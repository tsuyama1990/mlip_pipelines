# CYCLE05 SPEC: Dynamics Engine (LAMMPS & EON Integration)

## Summary
CYCLE05 implements the "Exploration" and "Inference" phases of the Active Learning loop via the `Dynamics Engine`. This engine is the bridge between the Orchestrator and the atomistic simulation software (LAMMPS for MD, EON for kMC). Its primary responsibility is to run simulations using the current MLIP and, crucially, to monitor the uncertainty ($\gamma$ value) of the predictions in real-time (On-The-Fly learning). If the uncertainty exceeds a safety threshold, the engine immediately halts the simulation, extracts the highly uncertain atomic structures, and returns control to the Orchestrator for further training (preventing unphysical simulation crashes).

## System Architecture
The components developed in this cycle reside in the `src/dynamics/` directory and implement the `AbstractDynamics` interface.

```text
src/
├── core/
│   ├── __init__.py           (AbstractDynamics interface)
│   └── exceptions.py         (DynamicsHaltInterrupt)
├── domain_models/
│   └── config.py             (DynamicsConfig, SystemConfig definitions)
└── dynamics/
    ├── **__init__.py**
    ├── **dynamics_engine.py**(Implements MDInterface via LAMMPS Python API/subprocess)
    ├── **eon_wrapper.py**    (Implements EONWrapper for kMC)
    └── **security_utils.py** (Path traversal prevention, environment validation)
```

**Key Interfaces:**
- `MDInterface`: Implements `AbstractDynamics` for classical Molecular Dynamics using LAMMPS.
- `EONWrapper`: Implements `AbstractDynamics` for kinetic Monte Carlo using EON.

## Design Architecture
The design prioritizes safety (Physics-Informed Robustness) and security (strict sandbox isolation).

1.  **Hybrid Potential Enforcement**: To guarantee physical safety during extrapolation, the `MDInterface` must automatically generate LAMMPS input scripts (`in.lammps`) that utilize `pair_style hybrid/overlay`. It must explicitly superimpose the ACE potential (`pace`) with the physical baseline (`zbl` or `lj`) defined in `SystemConfig.baseline_potential`.
2.  **Uncertainty Watchdog (`fix halt`)**: The `MDInterface` configures LAMMPS to continuously compute the extrapolation grade ($\gamma$) using `compute pace_gamma`. It sets a `fix watchdog` to halt the simulation if the maximum $\gamma$ exceeds `DynamicsConfig.uncertainty_threshold` (e.g., 5.0).
3.  **Halt Detection and Extraction**: When a simulation halts prematurely (detected via LAMMPS output logs or specific exit codes), the `extract_high_gamma_structures` method parses the final `dump.lammps` file. It identifies the specific atoms responsible for the high $\gamma$ value and extracts their local environments as ASE `Atoms` objects for the Oracle to evaluate.
4.  **EON Client Lifecycle Management**: EON processes (like `eonclient`) can easily become zombie processes if not managed carefully. The `EONWrapper` must explicitly manage the subprocess. It must monitor for a specific custom exit code (`100`) emitted by the EON potential driver (`pace_driver.py`) to indicate an extrapolation halt. If halted or if a general exception occurs, it must ensure the `eonclient` subprocess is forcefully terminated to prevent resource leaks.
5.  **Strict Sandbox Execution**: Because LAMMPS and EON involve running complex external binaries and reading potentially unsafe input files, `security_utils.py` ensures that all working directories are strictly resolved, absolute paths (`resolve(strict=True)`), forbidding symlinks or traversal attempts. All binary paths are validated against `DynamicsConfig.trusted_directories` and optional `binary_hashes`.

## Implementation Approach
1.  **Implement Security Utilities (`dynamics/security_utils.py`)**:
    -   Create functions to strictly canonicalize paths: `Path(v).resolve(strict=True)`. Reject paths containing `..` before resolution to prevent bypasses.
    -   Create functions to validate binaries against `trusted_directories` (ensuring ownership, non-world-writability, and non-symlink status). Check `binary_hashes` (SHA256) if provided in `DynamicsConfig`.
2.  **Implement MDInterface (`dynamics/dynamics_engine.py`)**:
    -   Create the `MDInterface` class inheriting from `AbstractDynamics`.
    -   Implement `run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]`:
        -   Generate `in.lammps` script. Include `pair_style hybrid/overlay pace zbl 1.0 2.0`.
        -   Include `compute pace_gamma all pace ... gamma_mode=1` and `fix watchdog all halt 10 v_max_gamma > 5.0 error hard`.
        -   Execute LAMMPS securely using `subprocess.run(shell=False)` with the validated `lmp_binary` path.
        -   Parse LAMMPS output/logs to determine if it completed successfully or halted due to the watchdog.
        -   Return a dict containing `halted` status and the `dump_file` path.
    -   Implement `extract_high_gamma_structures(self, dump_file: Path, threshold: float) -> list[Atoms]`:
        -   Read the LAMMPS dump file using ASE or custom parsing.
        -   Identify atoms where the $\gamma$ value (usually stored as an extra per-atom array) exceeds the `threshold`.
        -   Extract the full structure or a localized cluster centered on these atoms and return them as a list of `Atoms` objects.
3.  **Implement EONWrapper (`dynamics/eon_wrapper.py`)**:
    -   Create the `EONWrapper` class inheriting from `AbstractDynamics`.
    -   Implement `run_exploration` to generate `config.ini` based on `DynamicsConfig.eon_config_template`.
    -   Use `subprocess.Popen` to execute `eonclient`. Wait for completion or an explicit timeout.
    -   If `proc.returncode == 100`, log the OTF event, gracefully terminate the process, and return `{'halted': True}` along with the path to the high-$\gamma$ structure written by the driver. Ensure that `finally` blocks are used to `proc.kill()` the client if an unexpected error breaks the loop.

## Test Strategy

### Unit Testing Approach
-   **Security Utilities**: Test `security_utils.py` extensively. Create dummy directories and files using `tmp_path`. Verify that attempting to validate a path with `..` or a symlink outside the trusted directory raises a `ValueError`. Verify that a valid path is correctly canonicalized and returned.
-   **Hybrid Input Generation**: Instantiate `MDInterface` with a `DynamicsConfig` (specifying `zbl` baseline) and a mock potential path. Call the internal method responsible for generating `in.lammps`. Assert that the generated script string contains the exact line `pair_style hybrid/overlay pace zbl ...`.
-   **Dump Extraction**: Create a mock LAMMPS dump file containing per-atom $\gamma$ values, where one atom exceeds the threshold (e.g., $\gamma=6.0$) and others are below (e.g., $\gamma=1.0$). Call `extract_high_gamma_structures`. Verify that it correctly identifies the high-$\gamma$ atom and returns the corresponding `Atoms` object representing that structure.

### Integration Testing Approach
-   **Dynamics in Orchestrator**: In the orchestrator integration test, configure `DynamicsConfig`. Mock the `subprocess.run` call for LAMMPS, simulating a "halt" event (e.g., returning a specific error code or printing a "Halt triggered" message to a mock log file). Execute `run_cycle()`. Verify that the Orchestrator correctly detects the halt, extracts the mock structure from a dummy dump file, and proceeds to the DFT generation phase instead of terminating successfully.