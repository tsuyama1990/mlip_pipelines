# CYCLE06 UAT: Validator & Quality Assurance

## Test Scenarios

### Scenario 1: Quality Assurance Pass
**ID**: `UAT-06-01`
**Priority**: High
**Description**: Verify that the `Validator` correctly evaluates a high-quality machine learning potential against a held-out test dataset. If the Root Mean Square Error (RMSE) for both Energy and Forces is strictly below the strict thresholds defined in the `ValidatorConfig` (e.g., `0.002` eV/atom and `0.05` eV/A respectively), the potential must successfully pass the validation gate and be cleared for production deployment by the Orchestrator.

### Scenario 2: Quality Assurance Failure
**ID**: `UAT-06-02`
**Priority**: High
**Description**: Verify that the `Validator` correctly identifies and rejects an underperforming or physically unviable potential. If the calculated RMSE exceeds the defined thresholds, or if the potential fails critical physical stability checks (such as predicting non-zero forces for a perfect symmetric bulk crystal like BCC Fe), the validation must explicitly fail, preventing the Orchestrator from deploying the flawed potential.

### Scenario 3: Automated Report Generation
**ID**: `UAT-06-03`
**Priority**: Medium
**Description**: Verify that the `Reporter` component automatically generates a comprehensive HTML summary file (`validation_report.html`) summarizing the Quality Assurance metrics (RMSE values, stability status) regardless of whether the potential passed or failed the validation gate. This report provides crucial insight to the user regarding the performance of the generated MLIP.

## Behavior Definitions

### UAT-06-01: Quality Assurance Pass
```gherkin
GIVEN a `ValidatorConfig` with strict thresholds (`energy_rmse_threshold=0.002`, `force_rmse_threshold=0.05`)
AND a trained `potential.yace` file
AND a held-out test dataset `test.extxyz` containing known DFT ground truth data
WHEN the `Validator` evaluates the potential against the dataset (mocking the predictions to be highly accurate)
THEN the calculated Energy and Force RMSE values should be below the respective thresholds
AND the `Validator` should return a `ValidationResult` with `passed=True`
AND the Orchestrator should proceed to deploy the potential to the `potentials/` directory.
```

### UAT-06-02: Quality Assurance Failure
```gherkin
GIVEN a poorly trained `potential.yace` file
AND the `Validator` evaluates the potential against the test dataset (mocking the predictions to have high error)
WHEN the calculated Force RMSE (e.g., `0.10` eV/A) exceeds the configured `force_rmse_threshold` (e.g., `0.05` eV/A)
THEN the `Validator` should return a `ValidationResult` with `passed=False`
AND the `reason` should clearly state "Force RMSE exceeded threshold"
AND the Orchestrator should immediately halt the deployment process, preventing the flawed potential from entering production.
```

### UAT-06-03: Automated Report Generation
```gherkin
GIVEN the `Validator` has completed its evaluation and generated a `ValidationResult`
WHEN the `Reporter` processes this result
THEN a file named `validation_report.html` should be created in the current working directory
AND the file content should include the specific RMSE values calculated during the evaluation
AND the file content should clearly display the overall PASS/FAIL status in a human-readable format.
```