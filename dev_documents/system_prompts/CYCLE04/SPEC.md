# CYCLE04 SPEC: Trainer (Pacemaker Integration)

## Summary
CYCLE04 focuses on the "Refinement" phase of the Active Learning loop, implementing the `PacemakerWrapper` as the Trainer module. This component is responsible for processing the newly acquired DFT data from the Oracle, selecting the most informative structures using D-Optimality criteria (via `pace_activeset`), updating the accumulated dataset, and finally fine-tuning the Atomic Cluster Expansion (ACE) potential (via `pace_train`).

Crucially, this cycle enforces the "Physics-Informed Robustness" requirement by implementing Delta Learning. The Trainer configures Pacemaker to learn only the residual energy/forces above a predefined repulsive baseline (e.g., ZBL or LJ), ensuring that the resulting `.yace` potential never produces unphysical attractive forces at extremely short interatomic distances.

## System Architecture
The components developed in this cycle reside in the `src/trainers/` directory and implement the `AbstractTrainer` interface.

```text
src/
├── core/
│   └── __init__.py           (AbstractTrainer interface)
├── domain_models/
│   └── config.py             (TrainerConfig definition)
└── trainers/
    ├── **__init__.py**
    └── **ace_trainer.py**    (Implements PacemakerWrapper)
```

**Key Interfaces:**
- `PacemakerWrapper`: Implements `AbstractTrainer`. Responsibilities include `select_local_active_set` (D-Optimality), `update_dataset` (appending new data), and `train` (executing `pace_train` with Delta Learning constraints).

## Design Architecture
The design is strongly typed using the `TrainerConfig` Pydantic model to define strict boundaries for the Pacemaker CLI interactions.

1.  **Strict Subprocess Execution and Limits**: The `PacemakerWrapper` interacts with external Pacemaker binaries (`pace_train`, `pace_activeset`). To prevent command injection vulnerabilities, it uses `subprocess.run(shell=False)` with strictly constructed argument lists derived from the validated `TrainerConfig`. Crucially, to prevent infinite hangs and resource exhaustion during training (a common issue in iterative ML), `subprocess.run` must be wrapped with an explicit `timeout` argument.
2.  **D-Optimality Filtering**: Before adding new structures to the dataset, `select_local_active_set` uses `pace_activeset` to identify the most mathematically informative subsets from the candidates generated in CYCLE02. This prevents dataset bloat and over-representation of highly correlated structures, fulfilling the "Data Efficiency" requirement.
3.  **Delta Learning Configuration**: The `train` method strictly enforces the `baseline_potential` setting from `TrainerConfig` (e.g., "zbl"). It programmatically injects this requirement into the Pacemaker training configuration or CLI arguments, ensuring the ACE model learns only the many-body corrections to the strong repulsive core.

## Implementation Approach
1.  **Implement PacemakerWrapper (`trainers/ace_trainer.py`)**:
    -   Create the `PacemakerWrapper` class initialized with `TrainerConfig`.
    -   Implement binary path resolution and validation logic (reusing or adapting security utilities) to locate `pace_train` and `pace_activeset` securely.
    -   Implement `select_local_active_set(self, candidates: list[Atoms], anchor: Atoms, n: int = 5) -> list[Atoms]`:
        -   Write the `candidates` and the `anchor` structure to a temporary `extxyz` file.
        -   Construct the `pace_activeset` command list (e.g., `["pace_activeset", "-d", "temp.extxyz", "-n", str(n), "-a", "0"]` where index 0 is the anchor).
        -   Execute the subprocess securely.
        -   Parse the output indices and return the corresponding `Atoms` objects from the original list.
    -   Implement `update_dataset(self, new_atoms_list: list[Atoms], dataset_path: Path) -> Path`:
        -   Use ASE (`ase.io.write(..., append=True)`) to append the new, DFT-calculated `Atoms` objects to the accumulated `dataset_path` (e.g., `accumulated.extxyz`).
    -   Implement `train(self, dataset: Path, initial_potential: Path | None, output_dir: Path) -> Path`:
        -   Construct the `pace_train` command list using the `pace_train_args_template` from `TrainerConfig`.
        -   Inject the `dataset`, `max_epochs`, `active_set_size`, `baseline_potential`, and `regularization` parameters.
        -   If `initial_potential` is provided, add the `--initial_potential` argument for fine-tuning.
        -   Execute the subprocess securely (`shell=False`), explicitly passing `timeout=3600` (or a configurable value) and capturing stdout/stderr.
        -   If a `subprocess.TimeoutExpired` exception is caught, log the error and raise a `RuntimeError` to trigger the Orchestrator's general cleanup mechanism.
        -   Verify that the resulting `output_potential.yace` file exists in `output_dir` and return its path.

## Test Strategy

### Unit Testing Approach
-   **Secure Command Construction**: Instantiate `PacemakerWrapper` with a standard `TrainerConfig`. Mock the `subprocess.run` method. Call `train()`. Verify that the constructed command list passed to `subprocess.run` matches the expected format, contains no shell meta-characters (e.g., `;`, `&`, `|`), correctly incorporates the `baseline_potential` and `initial_potential` arguments, and that the `timeout` parameter is explicitly set.
-   **Dataset Updating**: Create a temporary file representing `accumulated.extxyz` containing one structure. Call `update_dataset` with a list of two new mock `Atoms` objects. Read the resulting file back using ASE and verify it now contains exactly three structures, demonstrating successful appending.

### Integration Testing Approach
-   **Trainer in Orchestrator**: In the orchestrator integration test, configure the `SystemConfig` and `TrainerConfig`. Mock the actual `pace_train` and `pace_activeset` subprocess calls to simply touch the expected output files (e.g., create a dummy `output_potential.yace`). Execute `run_cycle()`. Verify that the Orchestrator successfully passes the DFT-calculated structures to `self.trainer.update_dataset()`, calls `self.trainer.train()`, and successfully retrieves the mock potential file path for deployment. Ensure `tmp_path` is used for all operations.