# CYCLE06 Specification

## 1. Summary

CYCLE06 is the ultimate integration phase. It meticulously binds all the independent, refactored, and highly tested components from Cycles 1 through 5 into the continuous, fault-tolerant NextGen Orchestration loop. The central `Orchestrator` in `src/core/orchestrator.py` is entirely rewritten from the ground up to execute the 4-phase Hierarchical Distillation Workflow autonomously over weeks of execution.

Crucially, this cycle implements robust HPC (High-Performance Computing) capabilities. We introduce a local SQLite-based state management system (`checkpoint.py`). This guarantees that if a multi-day job is abruptly killed by a Slurm scheduler's wall-time limit, the Orchestrator can instantly and safely resume from the exact micro-operation (e.g., halfway through a massively parallel DFT batch or immediately prior to a LAMMPS resume command) without dropping any valuable quantum data. Furthermore, we implement an aggressive, asynchronous cleanup daemon. This daemon ensures that massive temporary files (like Quantum Espresso wavefunctions or gigabyte LAMMPS dumps) are aggressively deleted or compressed immediately after processing, preventing the pipeline from crashing due to filling the strict HPC storage quota limits.

## 2. System Architecture

The architecture fundamentally transitions from a simple, linear execution script to a highly robust, database-backed state machine capable of infinite looping and deep failure recovery.

The `Orchestrator` now relies heavily on the new `CheckpointManager` to persist its internal state before and after interacting with any heavy, failure-prone external domain module. The Orchestrator's `run_cycle` method becomes an infinite loop (or bounded loop based on max iterations configuration) that continually reads the database state, logically decides the next required phase (Distillation, Validation, Exploration, Extraction, Finetuning), and confidently dispatches the tasks to the relevant modules.

### File Structure Modification

We introduce a persistent database wrapper and heavily refactor the main loop to support state machine transitions rather than sequential function calls.

```text
.
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── **checkpoint.py**       (NEW: SQLite/JSON robust state management layer)
│   │   ├── **orchestrator.py**     (Major Refactor: Implements 4-Phase loop & Database State Machine)
```

By decoupling the state from the runtime memory, the Orchestrator itself becomes stateless and entirely resilient to sudden UNIX signals (`SIGKILL`, `SIGTERM`).

## 3. Design Architecture

The design implements the classic State Pattern combined with highly robust exception handling to manage the flow of control.

### Domain Concepts and Constraints

1. **Transaction Atomicity:** Every major pipeline step (e.g., "MACE Finetuning Completed", "DFT Batch 5 Completed", "LAMMPS Halt Parsed") must be written definitively to the SQLite database. If a Python process dies mid-step, the orchestrator on restart must read the last successfully committed transaction and re-execute only the failed step, never duplicating expensive DFT work.
2. **The 4-Phase State Machine:** The Orchestrator explicitly calls the `StructureGenerator` + `TieredOracle` for Phase 1. It explicitly calls the `Validator` for Phase 2. It then enters an inner `while True` loop calling the `DynamicsEngine`. Upon catching a custom `DynamicsHaltInterrupt`, it breaks out into Phase 3 (Cluster Extraction + DFT Oracle execution) and Phase 4 (Finetune MACE + ACE Incremental Update) before cleanly returning to the Phase 2 `while True` loop.
3. **Aggressive Artifact Cleanup:** The orchestrator must track the absolute paths of all generated large files (e.g., `.wfc`, `dump.lammps`, `.traj` files). After a transaction successfully commits, indicating those specific files are definitively no longer needed for that iteration, it must actively and safely call `os.remove` or `gzip` on them, catching and logging any permission errors without crashing the main loop.

## 4. Implementation Approach

The implementation focuses on creating a bulletproof database wrapper and undertaking massive control flow refactoring within the central brain of the application.

1. **Database Wrapper Creation:** Create `src/core/checkpoint.py`. Implement a simple but robust class utilizing Python's built-in `sqlite3` library. Create a master table `state (key TEXT PRIMARY KEY, value TEXT)`. Implement `set_state(key, value)` and `get_state(key)`. Use standard JSON serialization for storing complex dictionaries or lists. Ensure `isolation_level=None` to enable immediate autocommit behavior to survive sudden crashes.
2. **Orchestrator Init Refactoring:** Open `src/core/orchestrator.py`. In the `__init__` method, carefully initialize the `CheckpointManager`, the `TieredOracle`, the `FinetuneManager`, and all Pydantic configurations meticulously built in previous cycles. Establish the database connection immediately.
3. **The Master Run Loop:** Completely rewrite `run_cycle()`. Read the current state string from the database. Use a massive `if/elif` block or a State Pattern dispatch dictionary based on the stored state (e.g., `"PHASE1_DISTILLATION"`, `"PHASE3_MD_EXPLORATION"`). If no state exists, default to Phase 1.
4. **Halt Handling and Transitions:** Within the MD exploration block, wrap the `DynamicsEngine.run()` call in a strict `try/except` block specifically catching the custom `DynamicsHaltInterrupt`. In the `except` block, extract the high uncertainty atom indices, securely execute `extract_intelligent_cluster`, pass the resulting cluster to the `DFTManager`, trigger the hierarchical trainers, update the database state explicitly to `"PHASE3_MD_RESUME"`, and `continue` the massive loop.
5. **Cleanup Daemon Implementation:** Implement a `_cleanup_artifacts(paths: List[Path])` private method. Call this carefully at the very end of successful state phase transitions, explicitly ignoring `FileNotFoundError` to ensure idempotency.

## 5. Test Strategy

Testing this cycle requires complete, end-to-end simulation of the active learning pipeline using heavily mocked components to verify the state machine transitions accurately.

### Unit Testing Approach
We will thoroughly test the SQLite state management to ensure transaction integrity under duress.
- **Transaction Consistency Test:** Instantiate the `CheckpointManager` using a pure in-memory database (`sqlite:///:memory:`). Write complex nested JSON dictionaries representing the pipeline state. Retrieve them instantly. Assert absolute data integrity. Force an artificial exception mid-write and assert that the database rolls back to the previous consistent state correctly, preventing state corruption.
- **Cleanup Idempotency Test:** Call the cleanup method on a list of fake paths. Assert that the method simply logs the missing files and returns successfully, proving it will not crash the Orchestrator if a file was already deleted.

### Integration / E2E Testing Approach
We will execute a fully mocked "Dry Run" of the entire NextGen architecture to prove all components integrate flawlessly.
- **E2E State Machine Mocking:** We will instantiate the Orchestrator. We will inject `unittest.mock.MagicMock` instances for the `Generator`, `TieredOracle`, `DynamicsEngine`, and all `Trainers`. We will intelligently configure the `DynamicsEngine` mock to return normally on the first call, raise a precise `DynamicsHaltInterrupt` on the second call, and return normally on the third call to simulate a full active learning loop.
- **Workflow Verification:** We will execute `Orchestrator.run_cycle()`. We will mathematically assert that the mocks were called in the exact correct sequence mandated by the state machine (Generator -> Oracle -> Trainer -> Dynamics (OK) -> Dynamics (Halt) -> Extraction -> DFT -> Finetune -> Dynamics (Resume)). We will assert that the SQLite database correctly records these state transitions throughout the dry run.
- **Side-Effect Management:** The database is strictly in-memory. All massive file paths managed by the cleanup daemon point to Pytest `tmp_path` fixtures to guarantee perfectly clean execution without touching the local disk.
