# CYCLE06 User Acceptance Testing

## 1. Test Scenarios

**Scenario ID: UAT-C06-01**
**Priority: High**
**Title: HPC Wall-Time Job Kill Recovery and State Resumption**
This scenario strictly verifies that if the master Python orchestrator process is suddenly and violently terminated (e.g., simulating a Slurm scheduler killing a job upon hitting a strict 24-hour time limit), the system can seamlessly resume execution from the exact last successful micro-operation upon restart, without repeating hours of expensive DFT or MD calculations.

**Scenario ID: UAT-C06-02**
**Priority: High**
**Title: Execution of the Full 4-Phase Hierarchical Distillation Loop**
This scenario ensures that the orchestrator successfully and logically transitions through all four architectural phases: Zero-Shot Distillation, Validation, MD Exploration (encountering a Halt), Intelligent Cutout & DFT evaluation, Finetuning, and MD Resume. This proves the end-to-end integration of all prior developmental cycles.

**Scenario ID: UAT-C06-03**
**Priority: Medium**
**Title: Automated Artifact Cleanup and Quota Management**
This scenario confirms that the orchestrator's integrated cleanup daemon successfully deletes or compresses massive intermediate files (like Quantum Espresso `.wfc` wavefunctions and massive multi-gigabyte LAMMPS `.dump` files) immediately after they are successfully processed and committed. This critically prevents HPC storage quota breaches that would otherwise crash the pipeline.

**Scenario ID: UAT-C06-04**
**Priority: Low**
**Title: Handling Corrupted State Files**
This scenario ensures that if the SQLite database is somehow locked or corrupted by an external process, the Orchestrator fails loudly with a clear instruction to the user on how to recover, rather than silently overwriting data.

## 2. Behavior Definitions

**GIVEN** the orchestrator has successfully completed Phase 1 and is currently midway through processing the Phase 3 DFT batch
**WHEN** the python process is forcibly killed (via `SIGKILL`) and subsequently restarted by the user or scheduler
**THEN** the orchestrator must immediately read the SQLite checkpoint database
**AND** it must intelligently skip Phase 1 entirely and immediately resume the specific DFT batch calculation from the exact structure where it left off.

**GIVEN** a fully configured NextGen pipeline running in production mode
**WHEN** a `DynamicsHaltInterrupt` is correctly raised during the long-running MD exploration phase
**THEN** the orchestrator must systematically catch the error and call the `extract_intelligent_cluster` function
**AND** it must route the resulting pristine cluster to the `DFTManager` for evaluation
**AND** it must pass the resulting ground truth DFT data to the `FinetuneManager` and `ACETrainer`
**AND** it must finally call the `DynamicsEngine` to smoothly resume the simulation via `read_restart`.

**GIVEN** a successful, massive DFT Oracle execution that generated 50 GB of `.wfc` wavefunction files
**WHEN** the transaction is formally committed to the database indicating the data has been absorbed into the Replay Buffer and learned
**THEN** the cleanup daemon must automatically invoke `os.remove` on the `.wfc` files
**AND** a subsequent programmatic check of the file system must mathematically confirm their complete deletion.

**GIVEN** an active learning run where the SQLite checkpoint database file is locked by a read-only permission error
**WHEN** the Orchestrator initializes and attempts to write the starting state
**THEN** the system must immediately raise a critical file access exception
**AND** it must gracefully exit the program without attempting to run computationally expensive oracles.

## 3. Extended Behavior Verification

**GIVEN** a running loop completing its 50th continuous active learning cycle iteration
**WHEN** the Orchestrator prepares the next cycle's directories and state pointers
**THEN** the `CheckpointManager` must correctly log the iteration counter increment
**AND** the system must perfectly point the `DynamicsEngine` toward the `restart` file generated at the end of iteration 49.

**GIVEN** the pipeline catches a fatal Python exception in a deeply nested sub-module (e.g., `MemoryError` during Numpy array allocation)
**WHEN** the top-level Orchestrator `try/except` block engages
**THEN** the system must gracefully log the full stack trace to the terminal and standard output log file
**AND** it must critically update the `CheckpointManager` to a `"FAILED_FATAL"` state, halting execution entirely rather than infinitely retrying the broken process.
