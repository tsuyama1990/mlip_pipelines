# CYCLE04 UAT: Trainer (Pacemaker Integration)

## Test Scenarios

### Scenario 1: D-Optimality Active Set Filtering
**ID**: `UAT-04-01`
**Priority**: High
**Description**: Verify that the `PacemakerWrapper` successfully uses `pace_activeset` to select only the most mathematically informative structures from a large batch of redundant DFT calculations (candidates). This ensures the dataset remains compact and data-efficient, reducing training time while maintaining high accuracy. The anchor structure must always be included in the selection.

### Scenario 2: Physics-Informed Delta Learning configuration
**ID**: `UAT-04-02`
**Priority**: High
**Description**: Verify that the `PacemakerWrapper.train` method accurately constructs the `pace_train` command list, strictly enforcing the physics-informed baseline parameter (`baseline_potential`) defined in the `TrainerConfig`. This guarantees that the resulting MLIP will only learn the corrections to the strong core repulsive potential (like ZBL), preventing non-physical attractive forces at short distances.

### Scenario 3: Secure Subprocess Execution
**ID**: `UAT-04-03`
**Priority**: Medium
**Description**: Verify that the `PacemakerWrapper` securely executes external Pacemaker binaries by completely avoiding `shell=True` and using strictly parameterized command lists derived from the validated `TrainerConfig`. This prevents command injection vulnerabilities from potentially malicious `baseline_potential` or `dataset` path strings.

## Behavior Definitions

### UAT-04-01: D-Optimality Active Set Filtering
```gherkin
GIVEN a list of 20 highly correlated `Atoms` candidate structures
AND a single `Atoms` anchor structure `s0`
AND the `PacemakerWrapper` is configured with `n=5` for active set selection
WHEN `select_local_active_set()` is called (mocking the `pace_activeset` subprocess)
THEN the returned list of structures should contain exactly 5 unique `Atoms` objects
AND the anchor structure `s0` should be explicitly included in this list.
```

### UAT-04-02: Physics-Informed Delta Learning configuration
```gherkin
GIVEN a `TrainerConfig` specifying `baseline_potential="zbl"`
AND a valid `dataset.extxyz` file
AND an `initial_potential.yace` file
WHEN `train()` is called
THEN the constructed command list passed to `subprocess.run` (mocked) should include `--baseline_potential zbl`
AND it should include `--initial_potential initial_potential.yace`
AND it should include `--dataset dataset.extxyz`.
```

### UAT-04-03: Secure Subprocess Execution
```gherkin
GIVEN a `TrainerConfig` where a parameter contains potential shell meta-characters (e.g., `baseline_potential="zbl; rm -rf /"`)
WHEN `train()` attempts to construct and execute the `pace_train` command
THEN the command should be passed as a strictly sanitized list of strings to `subprocess.run(shell=False)`
AND the shell meta-characters should be treated as literal string arguments to the binary, preventing arbitrary code execution.
```