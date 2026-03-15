# CYCLE01 UAT: Core Orchestrator & Configuration

## Test Scenarios

### Scenario 1: Initial System Startup with Valid Configuration
**ID**: `UAT-01-01`
**Priority**: High
**Description**: Verify that the system successfully initializes and completes a simulated Active Learning cycle when provided with a valid `.env` configuration file defining basic elements and a strictly validated project root directory. The system should correctly execute the orchestration state machine (Exploration $\rightarrow$ Candidate Selection $\rightarrow$ DFT $\rightarrow$ Training $\rightarrow$ Deployment) using mock implementations, generating a valid (mocked) potential file without any real computational overhead.

### Scenario 2: Startup Failure on Invalid Configuration
**ID**: `UAT-01-02`
**Priority**: High
**Description**: Verify that the system immediately halts and reports clear validation errors when the provided `.env` configuration contains invalid settings, such as path traversal strings (`..`), unallowed directory permissions, or missing required fields. This ensures that the robust Pydantic configuration schemas (`ProjectConfig`, `SystemConfig`, etc.) are effectively guarding the system against misconfigurations that could lead to runtime failures or security vulnerabilities.

### Scenario 3: Secure Potential File Deployment
**ID**: `UAT-01-03`
**Priority**: Medium
**Description**: Verify that the Orchestrator's `_secure_copy_potential` method correctly validates the integrity and size of a newly generated MLIP (`.yace` file) before deploying it to the production directory. The test should simulate the generation of both a valid potential (correct headers, within size limits) and an invalid potential (missing headers, exceeding size limits), ensuring that the invalid file is rejected and the system cleans up the temporary directory correctly.

### Scenario 4: Automated State Checkpointing and Recovery
**ID**: `UAT-01-04`
**Priority**: High
**Description**: Verify that if the Orchestrator process is killed or preempted mid-run, it can seamlessly resume its iteration tracking upon restart. The system must autonomously scan the `potentials/` directory for the latest valid `.yace` file (e.g., `generation_008.yace`) and correctly initialize `self.iteration` (e.g., to 8) without requiring manual configuration changes from the user.

## Behavior Definitions

### UAT-01-01: Initial System Startup with Valid Configuration
```gherkin
GIVEN a valid `.env` configuration file exists specifying `MLIP_SYSTEM__ELEMENTS="['Fe', 'C']"` and a valid `MLIP_PROJECT_ROOT` path
AND the system is configured to use mock implementations for Dynamics, Oracle, and Trainer
WHEN the user initializes the Orchestrator with this configuration
AND the user triggers `run_cycle()`
THEN the Orchestrator should successfully execute the entire Active Learning loop
AND a new potential file `generation_001.yace` should be securely deployed to the `potentials/` directory
AND the temporary execution directories should be cleanly removed.
```

### UAT-01-02: Startup Failure on Invalid Configuration
```gherkin
GIVEN an invalid `.env` configuration file exists specifying an insecure `MLIP_PROJECT_ROOT` containing path traversal characters (`..`)
WHEN the user attempts to initialize the `ProjectConfig` with this file
THEN the initialization should fail immediately
AND a `pydantic.ValidationError` should be raised, clearly indicating that the path traversal characters are forbidden
AND the Orchestrator state machine should not be started.
```

### UAT-01-03: Secure Potential File Deployment
```gherkin
GIVEN the Orchestrator has completed the Training phase
AND a valid, mock `.yace` file (containing required headers and within size limits) exists in the temporary training directory
WHEN the Orchestrator attempts to deploy the potential via `_secure_copy_potential()`
THEN the validation should pass
AND the file should be atomically moved to `potentials/generation_XXX.yace`.

GIVEN the Orchestrator has completed the Training phase
AND an invalid, mock `.yace` file (exceeding the 100MB size limit) exists in the temporary training directory
WHEN the Orchestrator attempts to deploy the potential via `_secure_copy_potential()`
THEN the validation should fail
AND a `ValueError` should be raised indicating the file size limit was exceeded
AND the invalid file should not be deployed.

### UAT-01-04: Automated State Checkpointing and Recovery
```gherkin
GIVEN a previous run of the pipeline was preempted
AND the `potentials/` directory contains `generation_001.yace`, `generation_002.yace`, and `generation_003.yace`
WHEN the user initializes a new instance of the Orchestrator
THEN the Orchestrator should successfully execute `resume_state()`
AND `self.iteration` should be automatically set to `3`
AND the next generated potential should correctly be named `generation_004.yace`.
```
```