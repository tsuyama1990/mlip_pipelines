# Master Plan: User Acceptance Testing and Tutorials

## 1. Tutorial Strategy

The goal of the User Acceptance Testing (UAT) and tutorials is to provide a seamless, reproducible, and interactive experience for new users and developers evaluating the Adaptive-MLIP GUI backend. To achieve this, we will consolidate all UAT scenarios into a single, executable Python script utilizing the `marimo` framework.

This approach ensures that users can easily step through the entire system lifecycle—from initial configuration and "Run 0" validation to background Hyperparameter Optimization (HPO) and real-time active learning execution—without managing multiple fragmented test files or complex command-line arguments.

The tutorial will support a "Mock Mode" (default) designed for rapid Continuous Integration (CI) and local execution without requiring expensive GPU hardware or active API keys. In this mode, heavy machine learning operations (like MACE training) and lengthy LAMMPS molecular dynamics runs are automatically mocked using `unittest.mock` at the `sys.modules` level, allowing the entire notebook to execute in seconds. A "Real Mode" can be toggled by the user to execute the actual physics engines if the necessary binaries and hardware are present.

## 2. Tutorial Plan

We will create a single, comprehensive tutorial file:
**`tutorials/adaptive_mlip_gui_workflow.py`**

This descriptive and professional filename clearly indicates its purpose. The file will be a pure Python `marimo` notebook structured into the following sequential scenarios:

**Scenario 1: Intent-Driven Translation & Setup (Cycles 01 & 02)**
-   **Action:** Programmatically construct a JSON payload representing a user's high-level GUI selections (e.g., Target Material: "Pt-Ni", Accuracy vs. Speed: 1, and an array of Atom Indices to freeze).
-   **Verification:** Submit this payload to the mock FastAPI backend. Assert that the returned `ProjectConfig` contains the mathematically translated thresholds (e.g., `uncertainty_threshold: 0.15`) and that the semantic atom tags were correctly injected.

**Scenario 2: Auto-HPO Execution (Cycle 03)**
-   **Action:** Submit a "GENERALIZE" policy to the `/hpo/start` endpoint.
-   **Verification:** Poll the `/hpo/status` endpoint to retrieve the Pareto front of model candidates, demonstrating the asynchronous background task execution. Select "Model_A" and assert the active configuration is updated.

**Scenario 3: Pre-flight "Run 0" Diagnostics (Cycle 05)**
-   **Action:** Submit a deliberately malformed structure (e.g., colliding atoms) to the `/validate/run0` endpoint.
-   **Verification:** Assert that the backend instantly catches the infinite potential energy divergence and returns a clear, descriptive HTTP error payload, preventing the orchestrator from launching a doomed job.

**Scenario 4: Orchestrator Control & Telemetry (Cycles 04 & 05)**
-   **Action:** Establish a WebSocket connection to the mock `/ws/telemetry` endpoint. Submit a `START` command to `/orchestrator/command`.
-   **Verification:** Assert that the WebSocket client receives a stream of JSON-formatted `TelemetryPacket` objects containing mock training loss and MD energy data. Submit a `PAUSE` command, assert the Orchestrator halts gracefully, and finally submit a `RESUME` command to verify stateful continuation.

**Scenario 5: Dynamic Thermodynamic Tasks (Cycle 06)**
-   **Action:** Submit a payload configuring a GCMC task with specific Target Temperature and Pressure.
-   **Verification:** Assert that the backend's thermodynamic solver correctly calculates the required chemical potential ($\mu$) and that the generated LAMMPS script contains the highly complex `fix gcmc` block completely configured.

## 3. Tutorial Validation

To ensure the tutorial remains functional and does not rot as the codebase evolves, it will be strictly validated in the CI pipeline.

1.  **Execution Check:** The CI pipeline will execute the notebook as a standard Python script: `uv run python tutorials/adaptive_mlip_gui_workflow.py`. This ensures it runs linearly without the `marimo` server hanging the bash session.
2.  **Zero Side-Effects:** The script will wrap all operations in a `tempfile.TemporaryDirectory` and inject `sys.path.insert(0, os.getcwd())` to ensure local package resolution. Upon successful exit (Code 0), the script must leave zero residual HPC artifacts (like `.extxyz` databases or `log.lammps` files) on the host filesystem.
3.  **Strict Linting:** The tutorial file itself must pass all project-wide `ruff` and `mypy` checks (with necessary `# ruff: noqa` pragmas only where `marimo` injects uppercase fixture arguments).
