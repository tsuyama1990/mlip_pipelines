# CYCLE01 SPEC: Core Orchestrator & Configuration

## Summary
CYCLE01 establishes the foundational architecture of the high-efficiency Machine Learning Interatomic Potential (MLIP) construction system. The primary goal is to implement a robust, type-safe configuration layer and a secure orchestration state machine that manages the entire active learning loop (Exploration $\rightarrow$ Selection $\rightarrow$ Calculation $\rightarrow$ Refinement $\rightarrow$ Deployment). By leveraging strict Pydantic models for domain configurations (`SystemConfig`, `DynamicsConfig`, etc.) and abstract base classes for modularity (`AbstractDynamics`, `AbstractOracle`), we guarantee that the system operates safely, predictably, and with zero configuration ambiguity.

This cycle focuses heavily on boundary management and separation of concerns. The Orchestrator does not perform scientific calculations directly; instead, it delegates tasks to the respective abstract components while strictly managing the data flow, temporary directory isolation, and atomic file movements. Security is a paramount concern; the configuration models strictly validate environment variables, executable paths, and prevent path traversal vulnerabilities.

## System Architecture
The system architecture for CYCLE01 is centered around the `ProjectConfig` Pydantic model and the `Orchestrator` class. The Orchestrator consumes the validated configuration and injects it into the abstract domain components.

```text
src/
├── core/
│   ├── __init__.py           (Defines AbstractBaseClasses)
│   ├── exceptions.py         (Defines DynamicsHaltInterrupt, OracleConvergenceError)
│   └── **orchestrator.py**   (The central state machine)
├── domain_models/
│   ├── __init__.py
│   ├── **config.py**         (Strict Pydantic models for configuration)
│   └── **dtos.py**           (Data Transfer Objects like ExplorationStrategy)
```

**Key Interfaces:**
- `ProjectConfig`: The root configuration object, composed of nested, immutable Pydantic models (`SystemConfig`, `DynamicsConfig`, `OracleConfig`, `TrainerConfig`, `ValidatorConfig`).
- `Orchestrator`: The main class responsible for executing `run_cycle()`, managing the transition between AL phases, handling exceptions gracefully, and ensuring atomic updates to the potential files.

## Design Architecture
The system is fully designed around robust Pydantic-based schemas. The domain concepts represented in this cycle include:

1.  **Configuration Immutability and Strictness**: All configuration models (`ProjectConfig`, `SystemConfig`, `DynamicsConfig`, etc.) use `model_config = ConfigDict(extra="forbid")`. This ensures that any unrecognized configuration keys in the `.env` or `config.yaml` file immediately raise a validation error, preventing silent failures caused by typos.
2.  **Security and Path Validation**: The `ProjectConfig` includes extensive custom validators. `project_root` must be an absolute path, cannot contain traversal characters (`..`), and must not reside in restricted system directories (`/etc`, `/bin`). `.env` file loading is strictly secured; the file cannot be a symlink, must have secure permissions (not world-writable), and keys/values are checked against strict regular expressions to prevent environment injection.
3.  **Dependency Injection**: The Orchestrator is instantiated with the `ProjectConfig`. It then instantiates the concrete implementations of `AbstractDynamics`, `AbstractOracle`, `AbstractTrainer`, and `AbstractGenerator` by passing down the specific sub-configurations (`config.dynamics`, `config.oracle`, etc.).
4.  **Atomic File Operations**: The Orchestrator manages the Active Learning iterations via temporary directories (`tmp_work_dir`). The final potential file (`generation_NNN.yace`) is strictly verified for size limits (preventing OOM) and headers (preventing corruption) before being atomically moved to its final destination using `shutil.move()`.
5.  **Extensibility**: The configuration is designed to be easily extensible in future cycles. For example, adding new parameters to the Adaptive Policy Engine simply involves adding typed fields to the `PolicyConfig` model.

## Implementation Approach
1.  **Define Pydantic Models (`config.py`)**:
    -   Create `SystemConfig` defining `elements`, `baseline_potential`, and `restricted_directories`.
    -   Create `DynamicsConfig` defining parameters for LAMMPS/EON, including strict executable validation.
    -   Create `OracleConfig` defining parameters for DFT (Quantum Espresso).
    -   Create `TrainerConfig` defining parameters for Pacemaker.
    -   Create `ProjectConfig` encompassing all sub-configs, inheriting from `BaseSettings` to seamlessly read from `.env` files with the `MLIP_` prefix. Implement custom `_validate_env_key` and `_validate_env_value` methods.
2.  **Define DTOs (`dtos.py`)**:
    -   Create `ExplorationStrategy` to hold parameters like `md_mc_ratio` and `t_max`.
    -   Create `MaterialFeatures` to represent derived properties of the elements.
3.  **Define Abstract Interfaces (`core/__init__.py`)**:
    -   Create `AbstractDynamics`, `AbstractOracle`, `AbstractTrainer`, and `AbstractGenerator` with abstract methods representing their core responsibilities (e.g., `run_exploration`, `compute_batch`, `train`, `generate_local_candidates`).
4.  **Implement Orchestrator (`core/orchestrator.py`)**:
    -   Implement the `__init__` method to initialize the system based on `ProjectConfig`.
    -   Implement `run_cycle()`, the main loop. Use a robust context manager (`@contextlib.contextmanager`) with `try...finally` to ensure temporary execution directories (`tmp_work_dir`) are always cleaned up, even on failure.
    -   Implement `_secure_copy_potential()` to safely validate the resulting YACE file (size limit, header check) and atomically copy it to the `potentials/` directory.
    -   Handle domain-specific exceptions (`DynamicsHaltInterrupt`, `OracleConvergenceError`) appropriately.

## Test Strategy

### Unit Testing Approach
-   **Configuration Validation**: Instantiate `ProjectConfig` with various valid and invalid `.env` inputs using `monkeypatch.setenv()`. Verify that invalid paths (e.g., containing `..`), missing required fields, or unrecognized keys correctly raise `pydantic.ValidationError`. Verify that secure permission checks on the `.env` file work correctly.
-   **Atomic File Operations**: Mock the `shutil.move()` and `os.fstat()` calls in `Orchestrator._secure_copy_potential()`. Create a dummy `tmp_work_dir` using the `tmp_path` fixture. Verify that a valid mock `.yace` file is successfully verified and copied, while an invalid file (exceeding max size or missing headers) raises a `ValueError`.

### Integration Testing Approach
-   **Orchestrator Loop**: Create mock implementations of `AbstractDynamics`, `AbstractOracle`, and `AbstractTrainer` that simply return dummy success values without actually executing LAMMPS or DFT. Instantiate the `Orchestrator` with these mocks and a valid `ProjectConfig`. Call `run_cycle()` and verify that the orchestration state machine correctly transitions through all phases (Exploration $\rightarrow$ Candidate Selection $\rightarrow$ DFT $\rightarrow$ Training $\rightarrow$ Deployment) and successfully produces the final `generation_001.yace` file in the mocked output directory. Use `tmp_path` for all file I/O to ensure a side-effect-free execution environment.