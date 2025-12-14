# Audit Request: Cycle 3 Implementation (The Autonomous Loop)

**Auditor:** Antigravity
**Target Spec:** `SPEC-CYCLE3_FINAL.md`
**Context:** The Coder (Jules) claims to have finished Cycle 3. This includes the "Interactive LAMMPS Controller", "Real QE Oracle", and the "Orchestrator".

**Your Mission:**
Conduct a **CRITICAL CODE REVIEW** based on the Final Specification.
Do not trust the PR summary. Look at the logic.
If any "MVP Constraints" are violated or "Mocking Rules" are ignored, **REJECT** the PR immediately.

---

## 🔍 Audit Checklist

### 1. The LAMMPS Interface (`src/engines/lammps_mace.py`)
* **The "No NPT" Rule (Critical):**
    * Inspect the `_callback` function.
    * Verify that it **ONLY** writes forces (`f`) to LAMMPS.
    * **REJECT** if it attempts to pass Stress/Virial tensors (Cycle 3 MVP explicitly forbids this to avoid complexity).
* **Interrupt Logic:**
    * Verify that `UncertaintyInterrupt` is raised **inside the callback** when `u > threshold`.
    * Ensure this exception carries the `Atoms` object (so we can carve it later).
* **The Fix Command:**
    * Confirm it uses `fix external pf/callback`.

### 2. The QE Oracle (`src/oracles/qe_oracle.py`)
* **Cleanliness:**
    * Verify usage of `tempfile.TemporaryDirectory`.
    * **REJECT** if it runs `pw.x` in the current working directory (polluting the workspace).
* **Physics Automation:**
    * **Cutoffs:** Are `ecutwfc`/`ecutrho` derived dynamically from SSSP JSON? (Reject hardcoded values like 40/300 Ry).
    * **K-Points:** Is there logic to switch between Gamma-only (for clusters) and K-Mesh (for bulk)?

### 3. The Orchestrator (`src/core/orchestrator.py`)
* **Robustness:**
    * If `QeOracle` raises `OracleComputationError` (SCF failure), does the loop **continue** (discarding data) or **crash**? It MUST continue.
* **Flow:**
    * Verify the sequence: `Bootstrapping` -> `LammpsMaceDriver` -> `Interrupt` -> `BoxCarver` -> `Label` -> `Train`.

### 4. Unit Tests (`tests/unit/`)
* **Mocking Reality:**
    * Inspect `test_lammps_mace.py`. Does it import `lammps`?
    * **REJECT** if the test requires the `lammps` python module to be installed. It MUST use `unittest.mock.MagicMock`.
    * Inspect `test_qe_oracle.py`. Does it try to run `pw.x`? It MUST use mocks.
* **Import Errors:**
    * Verify that imports are `from engines...` NOT `from src.engines...` (The "src-layout trap").

---

## 🛑 Failure Criteria (Immediate Rejection)
1.  **NPT Attempted:** Code tries to handle virial stress in the Python callback.
2.  **Hardcoded Physics:** Cutoffs or K-points are fixed numbers.
3.  **Pollution:** QE output files (`*.out`, `*.wfc`) are left in the project root.
4.  **Environment Dependency:** Unit tests fail without `pw.x` or `lammps` installed.

**Output:**
Provide a structured Audit Report.
* ✅ **PASS** only if all criteria are met.
* ⚠️ **WARNING** for minor code style issues.
* ❌ **BLOCKER** for any violation of the above checklist.