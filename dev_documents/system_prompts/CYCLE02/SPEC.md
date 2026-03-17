# CYCLE02: GUI to LAMMPS Compile Layer (ASE Integration) Specification

## 1. Summary

The primary objective of CYCLE02 is to implement the "Visual and Semantic State Management" requirement defined in the PRD. Currently, configuring complex spatial regions in LAMMPS for simulations like surface reactions requires users to manually write error-prone scripts defining absolute geometric bounds (`region`, `group`). This cycle eliminates that requirement entirely, shifting the paradigm from text-based geometry to visual topology.

We will develop a highly robust translation layer that accepts visual, semantic tagging data originating from the frontend's 3D viewer (e.g., an array of atom indices representing the "frozen bottom layer" of a metallic slab). This integer array data will be securely encoded directly into an ASE (Atomic Simulation Environment) `Atoms` object using its built-in, highly efficient `tags` numpy array attribute. Subsequently, we will extensively refactor the existing `lammps_generator.py` module to recognize these specific ASE tags and automatically compile the corresponding, syntactically perfect LAMMPS command blocks (such as `region`, `group`, and `fix setforce 0.0`). This ensures that the user's visual intent, painted directly onto the 3D structure, perfectly matches the executed commands within the C++ physics engine, eradicating a massive source of human error.

This compilation approach provides incredible stability. Because the regions are defined by topological tags attached to specific atoms rather than static XYZ coordinate boxes, the system inherently supports severe structural deformations during high-temperature molecular dynamics without the designated regions "slipping" off their target atoms. This cycle effectively transforms the GUI from a simple data-entry wrapper into a true semantic compiler.

## 2. System Architecture

This cycle focuses on the critical data flow between the newly established API translation layer and the existing LAMMPS execution engine situated within the `src/dynamics` module. The architecture ensures that the frontend knows nothing about LAMMPS syntax, and LAMMPS knows nothing about the GUI, with ASE acting as the universal, strongly-typed intermediary.

**File Structure (ASCII Tree):**
```text
mlip-pipelines/
├── src/
│   ├── api/
│   │   ├── routes.py                   # Extended to cleanly accept semantic UI payloads
│   ├── domain_models/
│   │   ├── config.py                   # Extended DynamicsConfig with SemanticRegionConfig
│   │   └── gui_schemas.py              # New robust models for AtomSelections
│   ├── dynamics/
│   │   ├── dynamics_engine.py          # Updated to process and pass SemanticRegionConfig
│   │   └── **lammps_generator.py**         # Heavily extended to parse ASE tags into LAMMPS scripts
└── tests/
    └── dynamics/
        └── test_lammps_generator.py    # Comprehensive unit tests for visual-to-script translation
```

The frontend application will send an array of integer atom indices that the user has visually "painted" or selected in the 3D viewer. This array is caught and strictly validated by a new Pydantic model (`SemanticRegionConfig`), which ensures no out-of-bounds indices are present. Immediately prior to LAMMPS execution, the `dynamics_engine` maps these indices as integer tags (e.g., assigning tag `1` for all frozen atoms) into the base ASE `Atoms` object that represents the physical system. Finally, the string-generation logic within `lammps_generator.py` iterates over the `tags` array, dynamically generating the complex, multi-line blocks of text required by LAMMPS to enforce the user's visual intent, bypassing geometric region definitions entirely.

## 3. Design Architecture

**Domain Concepts & Pydantic Models:**
-   **`AtomSelectionConfig`**: A new, highly specific model residing within `gui_schemas.py`. It holds the precise visual selection intent from the user.
    -   `region_name`: String (e.g., "bottom_slab"). This string must be strictly validated to contain only alphanumeric characters and underscores, as it will be injected directly into a LAMMPS script variable.
    -   `indices`: List of Integers representing the specific 0-based atoms selected in the GUI.
    -   `action`: An Enum representing the semantic physical intent (e.g., `FREEZE` for zero force, `THERMOSTAT` for NVT control, `ACTIVE` for pure NVE).
-   **`SemanticRegionConfig`**: Seamlessly integrated into the core `DynamicsConfig` within `config.py`. It holds a complete list of `AtomSelectionConfig` objects, allowing multiple, distinct, and complex topological regions to be defined concurrently within a single simulation cell.

**Key Invariants, Constraints, and Validation Rules:**
1.  **Strict Index Bounds Checking**: The integer indices provided within any `AtomSelectionConfig` must be rigorously validated against the total number of atoms present in the actual `.extxyz` system structure *before* any `tags` are applied. Attempting to assign a tag to an index that exceeds the array size will result in fatal `IndexError` exceptions during ASE numpy manipulation. This is a critical security and stability constraint designed to prevent the orchestration loop from crashing midway through preparation.
2.  **Absolute Tag Conflict Resolution**: A single atom cannot possibly belong to two conflicting semantic regions simultaneously (e.g., it is physically impossible for an atom to be both absolutely `FREEZE`d and under active `THERMOSTAT` control). The Pydantic validation layer must compute set intersections and strictly reject any payloads containing overlapping index arrays, preventing contradictory commands from reaching LAMMPS.
3.  **LAMMPS Syntax Safety and Sanitization**: The strings generated for `region` and `group` names by the `lammps_generator.py` module must be strictly alphanumeric (plus underscores) to prevent LAMMPS parsing errors or script injection vulnerabilities. The `region_name` provided by the GUI must be aggressively sanitized via regex before insertion into the final script text.

## 4. Implementation Approach

The implementation requires careful orchestration of numpy arrays and precise string formatting to ensure the generated simulation scripts are flawlessly executable.

**Step 1: Pydantic Schema Definitions and Conflict Validation**
Define the `AtomSelectionConfig` and extend the `DynamicsConfig` with the optional `SemanticRegionConfig` wrapper. Implement a highly efficient `@model_validator` on `SemanticRegionConfig` that aggregates all provided lists of indices into Python `set` objects and checks for any intersections (overlaps). If an overlap is detected, it must immediately raise a `ValueError` detailing exactly which atom index was assigned multiple conflicting actions, halting the pipeline.

**Step 2: High-Performance ASE Tag Injection Logic**
Create a robust utility function within `src/dynamics/dynamics_engine.py` (e.g., `_apply_semantic_tags_to_atoms(atoms: ase.Atoms, config: SemanticRegionConfig)`). This function takes the initial ASE `Atoms` object and the validated `SemanticRegionConfig`. It first performs a strict bounds check, ensuring `max(indices) < len(atoms)`. Then, it initializes a blank tag array: `atoms.set_tags(np.zeros(len(atoms), dtype=int))`. It sequentially iterates through the `AtomSelectionConfig` list, mapping the Enum actions (e.g., `FREEZE` -> tag `1`) and assigning these integers directly to the underlying numpy `tags` array at the specified indices utilizing vectorized numpy indexing for maximum performance on structures with millions of atoms.

**Step 3: Extending the `lammps_generator.py` Compiler**
Significantly modify the core string generation logic within `lammps_generator.py`. Before appending the main NVE/NVT dynamics integration blocks, the generator must inspect the `atoms.get_tags()` array. If non-zero tags are present, it must dynamically construct the corresponding LAMMPS command strings.
For example, if tag `1` (representing FREEZE) is discovered on indices 0 through 10, the generator should bypass complex geometric bounding boxes. Instead, it must utilize the simpler, vastly more robust `group` command based explicitly on atomic IDs:
```lammps
group bottom_slab id 1:11
fix freeze_fix bottom_slab setforce 0.0 0.0 0.0
```
*Crucial Note:* The generator must automatically handle the translation from Python's 0-based indexing to LAMMPS's mandatory 1-based atomic ID indexing to prevent off-by-one errors that would silently corrupt the physical simulation.

**Step 4: Integration with Pre-flight Diagnostics**
Ensure that the newly generated semantic tags and resulting LAMMPS groupings are fully compatible with the system's Run 0 validation check, allowing any lingering syntax anomalies to be caught instantly before deploying to the cluster.

## 5. Test Strategy

Testing this cycle focuses intensely on the correct, zero-defect transformation of semantic intent into functional, syntax-error-free LAMMPS scripts, heavily utilizing regex string validation and numpy assertions.

**Unit Testing Approach:**
-   **Target:** The `_apply_semantic_tags_to_atoms` utility array manipulation.
-   **Method:** Instantiate a dummy 100-atom ASE object. Pass an `AtomSelectionConfig` defining the indices `[0, 1, 2]` with the action `FREEZE`. Assert using `numpy.testing.assert_array_equal` that `atoms.get_tags()` returns an array where exactly the first three elements are `1` and the remaining 97 elements are precisely `0`.
-   **Target:** Conflict Validation Logic.
-   **Method:** Construct and pass a `SemanticRegionConfig` where atom index `5` is present in both a `FREEZE` selection list and an `ACTIVE` selection list. Assert that a Pydantic `ValidationError` is instantly raised, proving the intersection logic is sound.
-   **Target:** Index Out of Bounds Security.
-   **Method:** Pass an index of `105` to a 100-atom simulated system. Assert a standard `ValueError` is raised well before any tags are ever applied to the object.

**Integration Testing Approach:**
-   **Target:** The entire `lammps_generator.py` string output pipeline.
-   **Method:** Create a highly controlled dummy ASE `Atoms` object with predefined tags manually injected (e.g., tag `1` applied strictly to indices `0-10`). Invoke the `lammps_generator.py` core generation function.
-   **Assert:** Perform strict regex matching on the returned multi-line script string. Verify categorically that the exact string sequence `group bottom_slab id 1:11` is present (proving the 1-based index conversion succeeded). Verify that `fix setforce 0.0 0.0 0.0` is correctly applied referencing that specific, sanitized group name.
-   **Side-effect Isolation:** Generate all LAMMPS scripts entirely in memory as strings. We will completely bypass executing the actual LAMMPS binary or writing temporary files during these unit tests, ensuring they run in sub-millisecond times and are highly portable across development environments.
