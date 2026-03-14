# CYCLE03 SPEC: Oracle (DFT Integration)

## Summary
CYCLE03 focuses on the "Calculation" phase of the Active Learning loop, implementing the `DFTManager` which serves as the Oracle. The Oracle is responsible for taking the localized, highly uncertain candidate structures generated in CYCLE02 and running high-fidelity Density Functional Theory (DFT) calculations (using Quantum Espresso via ASE) to determine the true ground-state energy, forces, and stresses.

A critical component of this cycle is the implementation of a robust self-healing mechanism to automatically recover from common DFT failures (e.g., SCF convergence errors) by dynamically adjusting parameters like mixing beta and smearing width. Furthermore, to prevent "surface effect" artifacts when calculating forces on small localized clusters, the Oracle implements a Periodic Embedding technique, wrapping the cluster in a periodic supercell before passing it to the DFT engine.

## System Architecture
The components developed in this cycle reside in the `src/oracles/` directory and implement the `AbstractOracle` interface.

```text
src/
├── core/
│   ├── __init__.py           (AbstractOracle interface)
│   └── exceptions.py         (OracleConvergenceError)
├── domain_models/
│   └── config.py             (OracleConfig definition)
└── oracles/
    ├── **__init__.py**
    └── **dft_oracle.py**     (Implements DFTManager and ASE Espresso Calculator)
```

**Key Interfaces:**
- `DFTManager`: Implements `AbstractOracle`. Responsibilities include `compute_batch` (running DFT on a list of structures), applying periodic embedding, and handling SCF retry logic.

## Design Architecture
The design is strongly typed using the `OracleConfig` Pydantic model to define strict boundaries for the DFT calculations.

1.  **DFT Configuration Constraints**: The `OracleConfig` enforces limits on the physical simulation box to prevent out-of-memory (OOM) errors and numerical overflow. Parameters like `max_atoms` (e.g., 10000), `max_cell_dimension`, and `kspacing` (e.g., between 0.01 and 0.15 $\text{\AA}^{-1}$) are strictly validated.
2.  **Self-Healing Calculator**: The `DFTManager.compute_batch()` method wraps the ASE `Espresso` calculator in a `try...except` block. If a calculation fails to converge within the defined `max_retries`, the manager automatically adjusts `mixing_beta` (e.g., from 0.7 down to 0.3) or changes the `diagonalization` algorithm (e.g., from 'david' to 'cg') before retrying. If it still fails, it raises a domain-specific `OracleConvergenceError`.
3.  **Periodic Embedding**: Instead of calculating properties on isolated vacuum clusters (which introduces surface artifacts), the `_apply_periodic_embedding()` method ensures that the structure retains periodic boundary conditions. It calculates an optimal orthorhombic bounding box based on the atoms' extent plus a defined `buffer_size` (e.g., 4.0 $\text{\AA}$), centers the atoms, and sets `pbc=True` before passing the modified `Atoms` object to the calculator.

## Implementation Approach
1.  **Implement DFTManager (`oracles/dft_oracle.py`)**:
    -   Create the `DFTManager` class initialized with `OracleConfig`.
    -   Implement `_apply_periodic_embedding(self, atoms: Atoms) -> Atoms`:
        -   Calculate the minimum and maximum coordinates of the input `atoms`.
        -   Calculate cell dimensions: `(max_coord - min_coord) + self.config.buffer_size * 2`.
        -   Validate that the new cell dimensions do not exceed `self.config.max_cell_dimension`.
        -   Create a new `Atoms` object with `pbc=True`, the calculated cell, and center the atoms within it.
    -   Implement `compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]`:
        -   Iterate through the `structures`.
        -   Apply `_apply_periodic_embedding()`.
        -   Set up the `ase.calculators.espresso.Espresso` calculator using parameters from `OracleConfig` (e.g., `kspacing`, `pseudopotentials`, `tprnfor=True`, `tstress=True`).
        -   Implement a retry loop (up to `self.config.max_retries`):
            -   Try `atoms.get_potential_energy()`.
            -   If an `ase.calculators.calculator.CalculationFailed` or similar exception occurs, adjust calculator parameters (e.g., reduce `mixing_beta`, change `diagonalization`).
            -   If successful, append the `atoms` object (now containing calculated forces/energies) to the results list and break the retry loop.
        -   If all retries fail, raise `OracleConvergenceError`.
        -   Return the successfully calculated `Atoms` objects.

## Test Strategy

### Unit Testing Approach
-   **Periodic Embedding Validation**: Instantiate `DFTManager` with a standard `OracleConfig`. Create a small, non-periodic `Atoms` object (e.g., a simple Fe dimer). Call `_apply_periodic_embedding()`. Assert that the returned `Atoms` object has `pbc=True`, that its cell dimensions correctly incorporate the `buffer_size`, and that the atoms are centered within the new cell. Create an artificially large structure exceeding `max_cell_dimension` and assert that a `ValueError` is raised.
-   **Self-Healing Logic**: Use `pytest-mock` to patch the ASE `Espresso.get_potential_energy()` method. First, simulate an immediate success and verify the structure is returned. Next, simulate a failure on the first attempt but a success on the second attempt. Verify that the `DFTManager` caught the first exception, modified the `mixing_beta` or `diagonalization` parameters on the calculator, and successfully returned the structure on the retry. Finally, simulate repeated failures exceeding `max_retries` and assert that an `OracleConvergenceError` is explicitly raised.

### Integration Testing Approach
-   **Oracle in Orchestrator**: In the orchestrator integration test, configure the `SystemConfig` and `OracleConfig`. Mock the ASE `Espresso` execution entirely (do not run actual `pw.x` binaries). Execute `run_cycle()`. Verify that the Orchestrator successfully passes the generated candidate structures to `self.oracle.compute_batch()`, that the batch is processed (and embedded), and that the resulting "calculated" structures are passed back to the Trainer for the next AL phase. Ensure `tmp_path` is used for the `calc_dir` to maintain isolation.