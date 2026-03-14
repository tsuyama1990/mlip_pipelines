# CYCLE06 SPEC: Validator & Quality Assurance

## Summary
CYCLE06 implements the "Validation" phase, representing the final Quality Assurance Gate before a newly trained MLIP is deployed into production by the Orchestrator. The `Validator` module rigorously tests the `.yace` potential against mathematical and physical criteria defined in the `ValidatorConfig`.

It evaluates numerical accuracy (Energy/Force RMSE on a held-out test dataset) and physical stability (e.g., verifying that a basic crystal structure satisfies Born stability criteria and does not exhibit imaginary phonon frequencies). Finally, the `Reporter` component generates a human-readable HTML summary (`validation_report.html`) detailing the test results.

## System Architecture
The components developed in this cycle reside in the `src/validators/` directory.

```text
src/
├── core/
│   └── orchestrator.py       (Calls Validator.validate())
├── domain_models/
│   └── config.py             (ValidatorConfig definition)
└── validators/
    ├── **__init__.py**
    ├── **reporter.py**       (HTML report generator)
    └── **validator.py**      (Implements Quality Assurance checks)
```

**Key Interfaces:**
- `Validator`: Evaluates a potential (`Path`) and returns a `ValidationResult` object indicating pass/fail status and specific metrics.
- `Reporter`: Consumes a `ValidationResult` and generates a static HTML file.

## Design Architecture
The design emphasizes deterministic, side-effect-free testing and strict configuration adherence.

1.  **Strict Validation Criteria**: The `ValidatorConfig` defines immutable thresholds (e.g., `energy_rmse_threshold=0.002`, `force_rmse_threshold=0.05`). The `Validator` must strictly adhere to these limits; if any metric exceeds its threshold, the validation fails.
2.  **Physical Stability Checks**: To ensure the potential describes a physically stable material, the `Validator` can programmatically build a reference crystal (e.g., BCC Fe, defined in `validation_crystal`, `validation_a`) using `ase.build.bulk`. It then uses external libraries (like `phonopy` or ASE's elasticity tools) to evaluate mechanical stability (e.g., calculating elastic constants to check Born criteria) or dynamical stability (phonon dispersion).
3.  **Deterministic RMSE Evaluation**: If a `test_dataset_path` is provided, the `Validator` uses the ASE `PaceCalculator` (or similar interface) to predict energies/forces for all structures in the dataset. It calculates the Root Mean Square Error (RMSE) between the DFT ground truth stored in the `extxyz` file and the MLIP predictions.
4.  **Reporting**: The `Reporter` simply takes the scalar metrics and boolean flags from the `ValidationResult` and formats them into an HTML string, writing it to the specified directory. It handles exceptions gracefully if the write fails, to prevent failing the entire cycle just for a reporting error.

## Implementation Approach
1.  **Implement Validator (`validators/validator.py`)**:
    -   Create the `Validator` class initialized with `ValidatorConfig`.
    -   Create a dataclass or Pydantic model `ValidationResult` containing: `passed: bool`, `reason: str`, `energy_rmse: float | None`, `force_rmse: float | None`, `is_mechanically_stable: bool | None`.
    -   Implement `validate(self, potential_path: Path) -> ValidationResult`:
        -   **Step 1: RMSE Evaluation**: If `config.test_dataset_path` exists, read it with ASE. Set the calculator for each structure to the `potential_path`. Calculate predictions, compute RMSE against true values, and check against `config.energy_rmse_threshold` and `config.force_rmse_threshold`.
        -   **Step 2: Stability Evaluation (Optional)**: Build the reference structure using `ase.build.bulk(config.validation_element, crystalstructure=config.validation_crystal, a=config.validation_a)`. Use ASE elasticity or a simplified heuristic to check if the structure is a local minimum (e.g., forces near zero) and if basic elastic constants are positive.
        -   **Step 3: Final Decision**: If all checks pass, set `passed=True`. Otherwise, set `passed=False` and populate `reason`.
2.  **Implement Reporter (`validators/reporter.py`)**:
    -   Create the `Reporter` class.
    -   Implement `generate_html_report(self, result: ValidationResult, report_path: Path) -> None`:
        -   Construct a basic HTML string injecting the values from `result`. Include color-coded status indicators (Green for PASS, Red for FAIL).
        -   Write the HTML string to `report_path`. Use `logging` to report success or catch `OSError`.

## Test Strategy

### Unit Testing Approach
-   **RMSE Calculation**: Instantiate `Validator` with a `ValidatorConfig` defining specific thresholds. Create a mock dataset file containing `Atoms` objects with known "true" energies/forces. Patch the `PaceCalculator` (or ASE calculator mechanism) to return specific predicted values that result in a known RMSE (e.g., one that passes, and one that fails). Assert that `validate()` returns `passed=True` and `passed=False` respectively.
-   **Reporter Formatting**: Instantiate `Reporter`. Pass a mock `ValidationResult` containing specific RMSE values and a `passed=True` status. Assert that the generated HTML file exists and contains the expected strings (e.g., the exact RMSE value, the word "PASS").

### Integration Testing Approach
-   **Validator in Orchestrator**: In the orchestrator integration test, configure the `ValidatorConfig`. Mock the internal `Validator.validate` method to simply return a passing `ValidationResult` without performing actual calculations. Execute `run_cycle()`. Verify that the Orchestrator successfully calls `self.validator.validate()`, receives the pass signal, calls the `Reporter` to write the HTML file in the `tmp_work_dir`, and successfully completes the deployment phase.