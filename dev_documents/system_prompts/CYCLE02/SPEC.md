# CYCLE02 SPEC: Structure Generator & Policy Engine

## Summary
CYCLE02 focuses on the "Exploration" phase of the Active Learning loop. Instead of relying on naive random sampling, the system implements an intelligent `AdaptiveExplorationPolicyEngine` that determines the optimal sampling strategy (e.g., Temperature schedules, MD vs. MC ratios, strain limits) based on the intrinsic properties of the material (e.g., metals vs. insulators). Additionally, the `StructureGenerator` is implemented to provide the physical candidate structures. It creates synthetic interfaces (like FePt/MgO) at initialization and generates localized perturbed candidate structures (via rattling) around highly uncertain atomic environments detected during the MD/kMC simulation.

## System Architecture
The components developed in this cycle reside in the `src/generators/` directory and implement the `AbstractGenerator` interface.

```text
src/
├── core/
│   └── __init__.py           (AbstractGenerator interface)
├── generators/
│   ├── **__init__.py**
│   ├── **adaptive_policy.py**(Implements AdaptiveExplorationPolicyEngine)
│   ├── **defect_builder.py** (Utilities for creating vacancies/interstitials)
│   └── **structure_generator.py**(Implements StructureGenerator and ASE generation)
```

**Key Interfaces:**
- `AdaptiveExplorationPolicyEngine`: Evaluates a material's features to output an `ExplorationStrategy`.
- `StructureGenerator`: Implements `AbstractGenerator`. Responsibilities include `generate_local_candidates` (rattling) and `generate_interface` (stacking bulk materials).

## Design Architecture
The design is heavily typed using Pydantic models from `src/domain_models/` to ensure robust inputs and outputs.

1.  **Adaptive Exploration Policy**: The policy engine (`adaptive_policy.py`) consumes the `PolicyConfig` (defining default scales and ratios) and `MaterialFeatures` (containing the elements and derived properties like predicted melting points). It outputs a strongly-typed `ExplorationStrategy` DTO containing precise numerical parameters for the next simulation run (e.g., `t_max`, `md_mc_ratio`).
2.  **Structure Generator Constraints**: The `StructureGenerator` consumes `StructureGeneratorConfig`. It is responsible for generating candidate structures using ASE (`ase.Atoms`). To prevent Denial of Service (DoS) via memory exhaustion, `generate_local_candidates` must strictly cap the number of generated structures (e.g., `n = min(n, 100)`) and the size of the input structure (e.g., `len(s0) < 10000`).
3.  **Interface Generation**: When configured via `InterfaceTarget` in the `SystemConfig`, the `StructureGenerator.generate_interface` method is triggered by the Orchestrator at a specific iteration (usually iteration 0). It uses `ase.build.bulk` and `ase.build.stack` to programmatically build the specified interface (e.g., stacking FCC FePt on Rocksalt MgO) instead of relying on external, potentially insecure template files. The requested elements are strictly validated against a whitelist (`valid_interface_targets`).

## Implementation Approach
1.  **Implement Adaptive Policy Engine (`generators/adaptive_policy.py`)**:
    -   Create the `AdaptiveExplorationPolicyEngine` class initialized with `PolicyConfig`.
    -   Implement the `decide_policy()` method. This method should contain the heuristic logic:
        -   If the material is a multi-component metal ($E_g \approx 0$), return a High-MC strategy ($R_{MD/MC}$ > 0).
        -   If it's a complex insulator ($E_g > 0$), return a Defect-Driven strategy ($N_{defects}$ > 0).
        -   If it's a hard material ($B_0$ is high), return a Strain-Heavy strategy ($\epsilon_{range}$ > 0).
        -   Default to a Cautious strategy (low $T_{max}$) if properties are highly uncertain.
2.  **Implement Structure Generator (`generators/structure_generator.py`)**:
    -   Implement the `StructureGenerator` class inheriting from `AbstractGenerator`.
    -   Implement `generate_local_candidates(self, s0: Atoms, n: int) -> list[Atoms]`:
        -   Enforce strict size limits on `s0` (e.g., reject if `> 10000` atoms) to prevent OOM.
        -   Use ASE's `rattle()` method to apply random Gaussian displacements to the atoms, controlled by the `stdev` parameter in `StructureGeneratorConfig`. Use a deterministic seed base for reproducibility.
        -   Yield or return a bounded list of up to `n` rattled `Atoms` objects.
    -   Implement `generate_interface(self, target: InterfaceTarget) -> Atoms`:
        -   Validate `target.element1` and `target.element2` against `config.valid_interface_targets` to prevent injection of unsupported materials.
        -   Use `ase.build.bulk` to create the two bulk materials.
        -   Use `ase.build.stack` to join them along the specified axis (e.g., `axis=2`). Apply necessary strain (`maxstrain=10.0`) to match lattice parameters. Return the combined `Atoms` object.

## Test Strategy

### Unit Testing Approach
-   **Policy Logic**: Instantiate `AdaptiveExplorationPolicyEngine` with a standard `PolicyConfig`. Pass various mocked `MaterialFeatures` (e.g., representing a metal, an insulator, a hard ceramic). Assert that `decide_policy()` correctly returns an `ExplorationStrategy` with the expected parameters (e.g., `md_mc_ratio > 0` for the metal, `n_defects > 0` for the insulator) corresponding to the defined heuristic rules.
-   **Structure Limits**: Instantiate `StructureGenerator`. Create a massive dummy `Atoms` object (e.g., 20,000 atoms). Call `generate_local_candidates` and assert that a `ValueError` is raised, confirming OOM protection. Create a valid, small `Atoms` object and assert that exactly `n` distinct rattled structures are returned.
-   **Interface Validation**: Call `generate_interface` with an invalid `InterfaceTarget` (e.g., elements not in the whitelist). Assert that a `ValueError` is raised, confirming injection protection. Call with valid elements (e.g., Fe and MgO) and assert that the resulting `Atoms` object contains the expected total number of atoms and the correct chemical symbols.

### Integration Testing Approach
-   **Generator in Orchestrator**: In the orchestrator integration test, configure `SystemConfig` with a valid `InterfaceTarget` and `interface_generation_iteration=0`. Mock the dynamics and training steps. Execute `run_cycle()`. Verify that the Orchestrator successfully calls `self.structure_generator.generate_interface()`, saves the initial structure to the working directory (`initial_structure.extxyz`), and proceeds without error. Use `tmp_path` to avoid writing to the real filesystem.