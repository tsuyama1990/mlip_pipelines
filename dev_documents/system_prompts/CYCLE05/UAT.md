# CYCLE05 User Acceptance Testing

## 1. Test Scenarios

**Scenario ID: UAT-C05-01**
**Priority: High**
**Title: Validation of Replay Buffer Historical Sampling and Prevention of Forgetting**
This scenario strictly verifies that the `ACETrainer` correctly maintains a historical database of explored bulk structures, and correctly samples from this massive database. This proves the newly trained potential will not suffer from "catastrophic forgetting" of previously stable phases when it learns about a new, highly distorted defect configuration.

**Scenario ID: UAT-C05-02**
**Priority: High**
**Title: Incremental Potential Update Configuration and Delta Learning Engagement**
This scenario ensures that the dynamically generated Pacemaker training configuration (`fit.yaml`) correctly points to the previous active learning iteration's potential file. This proves the system triggers a fast, incremental Delta Learning update rather than initiating a slow, computationally explosive from-scratch training run that would stall the HPC node for days.

**Scenario ID: UAT-C05-03**
**Priority: Medium**
**Title: Generation of MACE Finetuning CLI Arguments and Layer Freezing**
This scenario confirms that the `FinetuneManager` mathematically constructs the correct command-line string for the PyTorch execution. It must explicitly freeze the deep representation layers of the foundation model to preserve its critical zero-shot generalization capabilities across the broader chemical space.

**Scenario ID: UAT-C05-04**
**Priority: Low**
**Title: Robust Handling of Empty or Missing Replay Buffers on Iteration 1**
This scenario verifies that during the very first incremental update (Iteration 1), the system gracefully handles the fact that the replay buffer might be extremely small or nonexistent, defaulting to a physically safe behavior without raising array out-of-bounds exceptions during the random sampling phase.

## 2. Behavior Definitions

**GIVEN** a historical dataset residing on disk containing exactly 1000 structures and a configured `replay_buffer_size` of exactly 500
**WHEN** the `ace_trainer` receives a new surrogate dataset batch of 50 high-uncertainty structures
**THEN** the current training dataset compiled for Pacemaker must contain exactly 550 structures
**AND** the historical database file must be appended and updated to contain exactly 1050 structures.

**GIVEN** the orchestrator triggers an incremental update via the `ace_trainer` passing a previous potential path
**WHEN** the massive `fit.yaml` configuration file is generated
**THEN** the YAML file must contain the specific key `initial_potential` pointing absolutely to the previous generation's `.yace` file
**AND** the `max_num_epochs` parameter must be significantly lower (e.g., a factor of 10) than the Phase 1 cold-start default configuration.

**GIVEN** a batch of high-fidelity DFT structures passed to the `FinetuneManager` for adaptation
**WHEN** the `finetune_mace` subprocess command list is constructed
**THEN** the subprocess argument list must explicitly contain the `--freeze_body` flag (or its equivalent in the specific MACE CLI version)
**AND** it must point securely to the specific `mace_model_path` defined in the global system configurations.

**GIVEN** an active learning loop executing its first incremental update with an empty history file
**WHEN** the replay buffer sampling logic attempts to draw 500 structures
**THEN** the system must gracefully return all available structures without raising a `ValueError` for drawing a sample larger than the population
**AND** it must append the new DFT data to begin seeding the historical file for future iterations.

## 3. Extended Behavior Verification

**GIVEN** an active learning pipeline configured with `baseline_potential_type="ZBL"`
**WHEN** the Incremental Update is triggered and the training configuration is dumped
**THEN** the `fit.yaml` must dynamically contain the exact ZBL parameterizations for the specific elemental species present in the replay buffer
**AND** the underlying ACETrainer must completely avoid overwriting or ignoring this critical physical baseline.

**GIVEN** the orchestrator has passed the final updated potential back to the active directory
**WHEN** the `FinetuneManager` attempts to clean up its temporary HDF5 PyTorch datasets
**THEN** the system must mathematically assert that those temporary directories are completely removed from the `/tmp/` drive
**AND** it must correctly catch any `PermissionError` if the PyTorch process hasn't fully released the file lock, ensuring stability.
