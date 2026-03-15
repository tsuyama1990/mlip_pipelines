# User Acceptance Testing and Tutorial Master Plan

## Tutorial Strategy

The overarching goal of the User Acceptance Testing (UAT) is to amaze the user with the "Zero-Config" simplicity and power of the MLIP creation pipeline. To achieve this, the entire UAT suite and tutorial will be consolidated into a **single, interactive Marimo notebook**.

This approach provides a highly reproducible, visual, and educational experience. Users can execute the pipeline cell by cell, observing the Active Learning cycle (Exploration $\rightarrow$ Selection $\rightarrow$ Calculation $\rightarrow$ Training $\rightarrow$ Validation) unfold dynamically.

### "Mock Mode" vs. "Real Mode"
Because executing real DFT (Quantum Espresso) and MLIP training (Pacemaker) can take hours and require heavy, platform-specific dependencies, the tutorial must support two execution modes:

1.  **Mock Mode (Default/CI)**: Driven by the `MLIP_USE_MOCK=True` environment variable. In this mode, the heavy computational engines (ASE Espresso calculator, Pacemaker subprocesses, LAMMPS) are gracefully bypassed or patched via `unittest.mock`. The system will rapidly simulate the entire cycle—generating fake high-$\gamma$ structures, mocking successful DFT convergence, simulating a training run, and generating a dummy `.yace` potential. This allows the user (and CI pipelines) to verify the *orchestration* and *data flow* in seconds without installing the heavy backend tools.
2.  **Real Mode (Advanced)**: Driven by `MLIP_USE_MOCK=False`. In this mode, the system expects a fully configured HPC/workstation environment with `pw.x`, `pace_train`, `pace_activeset`, and `lmp` available in the path. The tutorial will run actual calculations. This mode is intended for users ready to construct their first real production potential.

## Tutorial Plan

A **single** Marimo Python file named `tutorials/UAT_AND_TUTORIAL.py` will be created. It will contain all scenarios (Quick Start + Advanced) organized linearly.

**Notebook Structure (`tutorials/UAT_AND_TUTORIAL.py`)**:

1.  **Introduction & Setup**:
    -   Displays the project logo/title using Markdown.
    -   Sets up the environment. By default, it will detect if heavy dependencies are missing and automatically enable `MLIP_USE_MOCK=True`.
    -   Initializes the `ProjectConfig` dynamically, demonstrating the "Zero-Config" aspect. The user only needs to specify `elements=["Fe"]` and `baseline_potential="zbl"`.

2.  **Scenario 1: Quick Start (The AL Loop)**:
    -   Instantiates the `Orchestrator` with the configuration.
    -   Executes `orchestrator.run_cycle()` in a loop (e.g., 3 iterations).
    -   Displays progress logs dynamically. In mock mode, this will complete in seconds, visually demonstrating the state transitions.

3.  **Scenario 2: Advanced - Interface Generation**:
    -   Demonstrates the `StructureGenerator`.
    -   Modifies the configuration to include an `InterfaceTarget` (e.g., `FePt` and `MgO`).
    -   Executes `orchestrator._pre_generate_interface_target()`.
    -   Visualizes the generated 3D atomic interface structure using ASE's built-in viewers (or `matplotlib`) directly within the Marimo cell.

4.  **Scenario 3: Validation and Reporting**:
    -   Executes the `Validator` on the final generated (or mocked) `.yace` potential.
    -   Displays the numerical results (Energy/Force RMSE).
    -   Renders the generated `validation_report.html` directly within the notebook using an `iframe`, proving the Quality Assurance gate works correctly.

## Tutorial Validation

To ensure the tutorial is robust and fully executable:

1.  **Headless CI Execution**: The CI pipeline will execute the notebook headlessly using `uv run marimo run tutorials/UAT_AND_TUTORIAL.py` with `MLIP_USE_MOCK=True`. This must pass without errors, proving the mock architecture and core logic are sound.
2.  **Strict Typing & Linting**: The notebook file itself will be subjected to strict `ruff` linting and `mypy` type checking (with specific exclusions for dynamic notebook features like `N803`, `B018` as defined in `pyproject.toml`).
3.  **Visual Inspection**: The final output of the notebook (when run interactively via `uv run marimo edit`) should cleanly render all markdown, logs, 3D structure visualizations, and the HTML report, confirming the user experience is polished and intuitive.