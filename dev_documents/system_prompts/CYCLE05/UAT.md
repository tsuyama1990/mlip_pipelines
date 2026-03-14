# CYCLE05 UAT: Dynamics Engine (LAMMPS & EON Integration)

## Test Scenarios

### Scenario 1: On-The-Fly (OTF) Extrapolation Halting
**ID**: `UAT-05-01`
**Priority**: High
**Description**: Verify that the `MDInterface` correctly configures LAMMPS to monitor the extrapolation grade ($\gamma$) during Molecular Dynamics simulations. If the model encounters an unknown atomic configuration and $\gamma$ exceeds the `uncertainty_threshold` (e.g., 5.0), the simulation must gracefully halt and return control to the Orchestrator for retraining, rather than continuing and potentially crashing due to unphysical forces.

### Scenario 2: Hybrid Potential Safety Enforcement
**ID**: `UAT-05-02`
**Priority**: High
**Description**: Verify that the `MDInterface` strictly generates LAMMPS input scripts that utilize a hybrid potential (`pair_style hybrid/overlay pace zbl`). This guarantees that the machine learning potential is always superimposed with a physical repulsive core, preventing atoms from overlapping non-physically even if the ACE model fails.

### Scenario 3: Secure Sandbox Path Execution
**ID**: `UAT-05-03`
**Priority**: Medium
**Description**: Verify that the `MDInterface` and `EONWrapper` strictly validate executable paths against `DynamicsConfig.trusted_directories` and optional `binary_hashes`. This ensures that malicious users cannot exploit the system by providing path traversal strings (e.g., `../../bin/sh`) or symlinks to execute unauthorized binaries on the host system.

## Behavior Definitions

### UAT-05-01: On-The-Fly (OTF) Extrapolation Halting
```gherkin
GIVEN a highly uncertain starting structure (mocked)
AND the `DynamicsConfig` sets `uncertainty_threshold=5.0`
WHEN the `MDInterface` executes the LAMMPS simulation (`run_exploration`)
THEN the generated `in.lammps` script should include `fix watchdog all halt 10 v_max_gamma > 5.0 error hard`
AND the (mocked) LAMMPS output indicating a halt should be correctly parsed by the interface
AND `run_exploration` should return a dictionary indicating `halted: True` along with the path to the final dump file.
```

### UAT-05-02: Hybrid Potential Safety Enforcement
```gherkin
GIVEN a `SystemConfig` specifying `baseline_potential="zbl"`
AND an `MDInterface` instance
WHEN the interface prepares the LAMMPS input script (`in.lammps`) for an exploration run
THEN the script must explicitly contain the line `pair_style hybrid/overlay pace zbl 1.0 2.0` (or similar parameters based on configuration)
AND it must contain the corresponding `pair_coeff` definitions for both `pace` and `zbl` for all elements in the system.
```

### UAT-05-03: Secure Sandbox Path Execution
```gherkin
GIVEN a `DynamicsConfig` with `trusted_directories=["/opt/mlip_bin"]`
AND an `lmp_binary` configuration maliciously set to `../usr/bin/python3`
WHEN the `MDInterface` attempts to validate and execute the binary
THEN the security utilities should detect the path traversal attempt (`..`)
AND a `ValueError` should be raised immediately
AND the LAMMPS subprocess should not be executed.
```