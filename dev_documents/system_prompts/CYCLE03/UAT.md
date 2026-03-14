# CYCLE03 UAT: Oracle (DFT Integration)

## Test Scenarios

### Scenario 1: Successful DFT Calculation and Embedding
**ID**: `UAT-03-01`
**Priority**: High
**Description**: Verify that the `DFTManager` correctly embeds a localized, non-periodic candidate cluster into an orthorhombic supercell before passing it to the Quantum Espresso calculator. This ensures that the generated training data is free from surface artifacts and includes accurate Hellmann-Feynman forces and stresses within a periodic bulk environment.

### Scenario 2: Self-Healing on SCF Convergence Failure
**ID**: `UAT-03-02`
**Priority**: High
**Description**: Verify that the Oracle's self-healing mechanism automatically catches standard DFT convergence errors (e.g., SCF non-convergence) and retries the calculation with dynamically adjusted, more stable parameters (like a lower `mixing_beta` or a different `diagonalization` algorithm). This prevents the entire Active Learning pipeline from crashing due to a single difficult structural configuration.

### Scenario 3: Exceeding Physical Constraints
**ID**: `UAT-03-03`
**Priority**: Medium
**Description**: Verify that the Oracle strictly enforces physical dimension limits (`max_cell_dimension`) and atom counts (`max_atoms`) defined in the `OracleConfig` to prevent memory exhaustion (OOM) attacks or numerical overflows when embedding excessively large candidate clusters.

## Behavior Definitions

### UAT-03-01: Successful DFT Calculation and Embedding
```gherkin
GIVEN a list of localized `Atoms` objects representing highly uncertain clusters
AND the `DFTManager` is configured with a `buffer_size` of 4.0 Å
WHEN `compute_batch()` is called with these structures
THEN each structure should be processed by `_apply_periodic_embedding()`
AND the resulting `Atoms` objects passed to the calculator should have `pbc=True`
AND the new cell dimensions should accurately encompass the original cluster plus the 4.0 Å buffer on all sides
AND the atoms should be centered within this new periodic cell.
```

### UAT-03-02: Self-Healing on SCF Convergence Failure
```gherkin
GIVEN a candidate structure that is known to cause SCF convergence issues (mocked for testing)
AND the `DFTManager` is configured with `max_retries=3`
WHEN `compute_batch()` executes the first calculation attempt
THEN the calculator (mocked) should raise an `ase.calculators.calculator.CalculationFailed` error
AND the `DFTManager` should catch this exception
AND the `mixing_beta` parameter on the calculator should be automatically reduced (e.g., from 0.7 to 0.3)
AND the `DFTManager` should automatically retry the calculation
AND the second attempt (mocked to succeed) should successfully return the structure with calculated forces and energies.
```

### UAT-03-03: Exceeding Physical Constraints
```gherkin
GIVEN a massive candidate cluster where the maximum interatomic distance exceeds the `max_cell_dimension` (e.g., > 1000 Å)
WHEN `_apply_periodic_embedding()` attempts to create a supercell for this structure
THEN the validation logic should immediately detect the violation
AND a `ValueError` should be raised, indicating the cell dimension is too large
AND the structure should be safely skipped or the batch calculation should halt, preventing OOM crashes.
```