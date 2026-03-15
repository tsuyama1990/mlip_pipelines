# Master Plan for User Acceptance Testing and Tutorials

## 1. Tutorial Strategy

The overarching goal of the PYACEMAKER NextGen tutorial strategy is to provide a seamless, reproducible, and interactive learning experience for users ranging from experimental material scientists to computational physicists.

To achieve this, we will transition the abstract user scenarios into executable, pure Python tutorials. These tutorials will leverage the `marimo` framework. While standard Jupyter Notebooks often suffer from hidden state and out-of-order execution issues, Marimo guarantees a reactive, acyclic execution graph. This ensures that the user's workspace remains perfectly consistent, making the tutorials exceptionally robust and suitable for automated CI/CD testing.

The tutorials will support two distinct execution modes to cater to different environments:
- **Mock Mode (CI / No-GPU Execution):** By default, or when specific environment variables (e.g., `PYACEMAKER_MOCK_MODE=1`) are set, the tutorial will instantiate "Dummy" versions of the heavy Oracle and Dynamics modules. This allows users to quickly verify the pipeline logic, configuration parsing, and system architecture without requiring an API key, a GPU, or hours of compute time.
- **Real Mode:** When executed on a proper High-Performance Computing node with the required backend dependencies (MACE, Quantum Espresso, LAMMPS) installed, the tutorial will run the actual physical simulations, generating real `potential.yace` files and executing true active learning loops.

## 2. Tutorial Plan

We will consolidate all critical use-cases (Quick Start configuration, Phase 1 Distillation, and Phase 3 Active Learning Interrupts) into a **SINGLE** comprehensive Marimo Python file. This file will be logically divided into cells representing the distinct phases of the workflow.

**Target File:** `tutorials/FePt_MgO_interface_energy.py`

This single tutorial will guide the user through the process of building an active learning potential capable of simulating the complex metallic-oxide interface between FePt and MgO.

### Tutorial Sections (Marimo Cells):

1. **Introduction and Imports:** Loading the PYACEMAKER orchestrator, configurations, and visualization utilities.
2. **Configuration Definition:** Demonstrating how to construct the nested Pydantic configuration (`ProjectConfig`, `ActiveLearningThresholds`, `DistillationConfig`, `CutoutConfig`). The tutorial will programmatically define these configurations to avoid requiring separate YAML files.
3. **Phase 1: Zero-Shot Distillation:** Initializing the `StructureGenerator` and `TieredOracle`. Demonstrating how the MACE foundation model acts as a surrogate oracle to quickly build the baseline ACE potential.
4. **Phase 2 & 3: Exploration and Intelligent Cutout:** Simulating the `DynamicsEngine` loop. The tutorial will intentionally trigger an "AL_HALT" event (by passing a highly distorted interfacial structure), and visually demonstrate the `extract_intelligent_cluster` function in action. It will plot the extracted core and the passivated buffer region.
5. **Phase 4: Hierarchical Finetuning:** Showing the data flow from the `DFTManager` back into the `FinetuneManager` and `ACETrainer`, updating the potential incrementally.
6. **Validation and Reporting:** Running the `Validator` to generate the final parity plots and phonon dispersion curves, proving the newly updated potential's physical accuracy.

## 3. Tutorial Validation

To ensure the highest quality, the tutorial must be systematically validated.

- **Executable Verification:** The tutorial file (`tutorials/FePt_MgO_interface_energy.py`) must be executed as a standard Python script (`uv run python tutorials/FePt_MgO_interface_energy.py`) during the final integration pipeline. It must execute from top to bottom without hanging or raising unhandled exceptions in Mock Mode.
- **Artifact Generation:** The tutorial must successfully generate its expected outputs (e.g., a dummy `generation_001.yace` file, a validation report HTML, and temporary `log.lammps` outputs) within a designated `tutorials/output/` directory, proving that the file I/O operations are functioning correctly.
- **Code Quality:** The tutorial script must adhere to the project's strict Ruff and MyPy linting standards, though specific exemptions (like `T201` for `print` statements) will be explicitly permitted within the `pyproject.toml` to allow for interactive user feedback.
