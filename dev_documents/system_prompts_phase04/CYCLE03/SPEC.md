# CYCLE03 Specification

## 1. Summary

CYCLE03 introduces the paradigm-shifting concept of Foundation Models (e.g., MACE-MP-0) into the Active Learning pipeline. In Phase 01 of the project, the system relied entirely on extremely expensive Density Functional Theory (DFT) calculations to evaluate every single atomic structure generated during the initial exploration phase. Because DFT scales factorially with the number of electrons, this purely quantum-mechanical approach severely bottlenecked the speed of the active learning loop, severely limiting the breadth of the chemical, structural, and thermodynamic space that could be explored within a reasonable HPC timeframe.

To definitively overcome this critical computational bottleneck, we implement the `Tiered Oracle` and `MACEManager` modules within the `src/oracles/` directory. This enables the "Zero-Shot Distillation" workflow. The system uses the MACE foundation model as an ultra-fast surrogate oracle to evaluate tens of thousands of generated structures in mere seconds. Crucially, the system not only extracts the predicted energies and forces from MACE, but also rigorously extracts the epistemic uncertainty metric. If the foundation model is highly confident in its prediction (i.e., the uncertainty falls strictly below the `threshold_call_dft` defined in CYCLE01), the structure and its MACE-predicted forces are immediately accepted into the Phase 1 training set. Only if MACE is uncertain, indicating novel physics outside its vast pre-training dataset, does the `TieredOracle` intelligently route the structure to the heavy `DFTManager` for high-fidelity, first-principles evaluation. This dramatically improves data efficiency without sacrificing the absolute physical correctness of the resulting ACE potential.

## 2. System Architecture

The architecture introduced in this cycle fundamentally alters how the Orchestrator interacts with computational engines. It introduces a new, powerful layer of abstraction over all Oracles. The existing `DFTManager` implicitly implemented a conceptual Oracle interface; this cycle formalizes that contract into a strict `BaseOracle` abstract base class. We then implement two new concrete classes: `MACEManager` (which implements the interface using the PyTorch-based MACE neural network) and `TieredOracle` (which implements the interface but acts primarily as a dynamic, uncertainty-driven router).

The central `Orchestrator` will no longer instantiate the heavy `DFTManager` directly during Phase 1. Instead, it instantiates the `TieredOracle`, passing it instances of both the `MACEManager` (as the primary, fast oracle) and the `DFTManager` (as the fallback, slow oracle). When the `Orchestrator` calls the unified `compute_batch` method with thousands of structures, the `TieredOracle` handles the complex orchestration internally. It sequentially queries MACE, filters the results based on the configuration threshold, routes the highly uncertain remainder to the DFT queue, manages the parallel DFT execution, and finally returns a unified, physically coherent dataset to the Orchestrator. This ensures the Orchestrator remains ignorant of the complex underlying routing logic, maintaining clean architecture.

### File Structure Modification

The file structure is expanded to accommodate the new foundation model wrappers and routing logic, sitting cleanly alongside the legacy DFT code.

```text
.
├── src/
│   ├── oracles/
│   │   ├── __init__.py
│   │   ├── base.py                 (NEW: Formal Abstract Base Class for all Oracles)
│   │   ├── dft_oracle.py           (Existing: Adapted to inherit from BaseOracle)
│   │   ├── **mace_manager.py**     (NEW: High-performance wrapper for MACE foundation model)
│   │   ├── **tiered_oracle.py**    (NEW: Dynamic routing logic between MACE and DFT based on uncertainty)
```

By placing `base.py` here, we enforce the Liskov Substitution Principle. Any module expecting a `BaseOracle` can seamlessly accept the `TieredOracle`, the `MACEManager`, or the `DFTManager` without changing a single line of consuming code.

## 3. Design Architecture

The design architecture centers around the polymorphic execution of Oracles and the strict standardization of physical metrics—specifically energy, force, and uncertainty—within the `ase.Atoms` object dictionaries.

### Domain Concepts and Constraints

1. **`BaseOracle` Interface:** This is an Abstract Base Class (ABC) defining the mandatory method signature: `compute_batch(structures: List[Atoms], work_dir: Path) -> List[Atoms]`. This enforces structural contract adherence across all Oracle implementations. Every oracle must accept a list of raw structures and return a list of structures annotated with the calculated physical properties.
2. **`MACEManager`:** A highly specialized Oracle utilizing the `mace` Python package. It must dynamically load the massive PyTorch model specified in `DistillationConfig.mace_model_path`. The core invariant is that after `compute_batch` is executed, every returned `Atoms` object MUST have its `info['energy']` (float), `arrays['forces']` (Numpy array), and critically, `info['mace_uncertainty']` (float) populated correctly. If the model fails to predict uncertainty, it must raise a loud, specific exception rather than silently passing null values.
3. **`TieredOracle`:** Acts as the intelligent composite Oracle. It requires initialization with instances of a `primary_oracle` (MACE), a `fallback_oracle` (DFT), alongside the scalar `threshold_call_dft` float. Its responsibility is purely logical routing; it must not perform physics calculations itself.
4. **Uncertainty Standardization:** MACE provides uncertainty via different mathematical mechanisms (e.g., committee variance or latent space Mahalanobis distances). This raw mathematical metric must be carefully normalized or standardized by the `MACEManager` so that it directly compares apples-to-apples against the user-defined `threshold_call_dft` float defined in the CYCLE01 configuration. This is crucial for predictable active learning behavior.

## 4. Implementation Approach

The implementation focuses on creating robust, fault-tolerant Python wrappers around complex PyTorch models and ensuring the pure routing logic is perfectly deterministic.

1. **Base Interface Definition:** Create `src/oracles/base.py`. Implement the `BaseOracle` ABC with the `compute_batch` abstract method. Ensure the existing `DFTManager` is refactored to inherit from this interface, fixing any mismatched method signatures.
2. **`MACEManager` Implementation:** Create `src/oracles/mace_manager.py`. Import the `mace.calculators.mace_mp` calculator (or equivalent). Implement the `compute_batch` loop. For each `Atoms` object, attach the loaded MACE calculator. Call `get_potential_energy()`, `get_forces()`, and crucially, extract the specific uncertainty metric (e.g., via the variance of the ensemble if utilizing MACE-MP-0). Ensure these values are carefully written into the `Atoms` dictionary structures to prevent overriding existing tags unnecessarily.
3. **`TieredOracle` Implementation:** Create `src/oracles/tiered_oracle.py`. Instantiate with `primary_oracle`, `fallback_oracle`, and `threshold`. In `compute_batch`, first pass the entire list of incoming structures directly to `primary_oracle.compute_batch()`.
4. **Routing Logic:** Iterate through the annotated results returned from the primary oracle. Extract `mace_uncertainty = atoms.info.get('mace_uncertainty', float('inf'))`. If this uncertainty is strictly less than the `threshold`, append the structure directly to the `final_results` list. If it meets or exceeds the threshold, or if the uncertainty is missing entirely, append the original, uncalculated structure to a separate `fallback_queue`.
5. **Fallback Execution:** Pass the entire `fallback_queue` to the `fallback_oracle.compute_batch()`. Extend the `final_results` list with these high-fidelity, rigorously DFT-calculated structures. Return the combined, unified list to the caller.

## 5. Test Strategy

Testing this cycle requires carefully mocking the heavy, multi-gigabyte MACE PyTorch models and strictly verifying the batch routing logic under edge cases.

### Unit Testing Approach
We will thoroughly test the pure `TieredOracle` routing logic using fast, isolated mock oracles within `tests/oracles/test_tiered_oracle.py`.
- **Routing Threshold Test:** We will create a `MockPrimaryOracle` that deterministically returns a list of 10 `Atoms` objects, where exactly 5 have an injected uncertainty of `0.01` and 5 have an injected uncertainty of `0.10`. We will set the `TieredOracle` threshold to `0.05`. We will create a `MockFallbackOracle` that simply flags structures it receives. We will assert mathematically that exactly 5 structures are passed to the `MockFallbackOracle`, and that the `TieredOracle.compute_batch()` method returns exactly 10 structures in total, proving the routing logic split the batch perfectly.
- **Missing Uncertainty Test:** We must ensure that if the primary oracle inexplicably fails to populate the `mace_uncertainty` key, the `TieredOracle` defaults to a mathematically safe behavior (e.g., routing everything to the fallback oracle to prevent learning garbage, unchecked data). We will pass structures lacking the key and assert the fallback oracle is called for all of them.

### Integration Testing Approach
We must test the `MACEManager` without blowing up CI memory limits.
- **MACE Calculator Integration:** We will write a test that initializes `MACEManager` with a specific `mace_model_path`. Instead of loading a real model, we will use `unittest.mock.patch` to intercept the `mace.calculators` import entirely, injecting a lightweight mock that mimics the ASE Calculator interface. We will assert that the calculator is properly attached to the ASE atoms and that the specific dictionary extraction of uncertainty from the MACE output dictionary functions correctly without raising missing key errors or type errors.
- **Side-Effect Management:** The MACE model loading requires significant disk I/O and RAM allocation. Our unit tests will strictly and completely mock the `MACEManager` PyTorch instantiation to prevent loading tensors during CI runs. All file I/O related to DFT fallback tests will be strictly confined using Pytest's `tmp_path` fixture.
