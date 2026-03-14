# MLIP Pipelines

Automated Machine Learning Interatomic Potential (MLIP) Builder.

This zero-configuration, autonomous active learning pipeline seamlessly integrates dynamic exploration (MD/kMC) with high-fidelity *ab initio* calculations (DFT) to train, evaluate, and deploy Advanced Configuration Exploration (ACE) potentials without manual intervention.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

## Key Features

*   **Zero-Configuration Active Learning:** Start a completely autonomous pipeline (Explore $\rightarrow$ Halt $\rightarrow$ DFT $\rightarrow$ Train) with just a single declarative YAML file.
*   **On-The-Fly (OTF) Self-Healing:** The system actively monitors extrapolation grades ($\gamma$) during LAMMPS/EON simulations. It safely halts before unphysical crashes occur, computes precise ground truth forces for uncertain structures using Quantum ESPRESSO via ASE, and seamlessly resumes dynamics.
*   **Targeted Interface Generation:** Autonomously builds complex structural boundaries (e.g., FePt/MgO) utilizing adaptive structure generators and D-Optimality active set selections to minimize computational overhead.
*   **Strict Security & Immutability:** Fully sandboxed subprocess execution via temporary isolated directories, strict configuration validation against path traversal vulnerabilities, and immutability guarantees through robust Python Pydantic models.

## Architecture Overview

The pipeline strictly adheres to a "Hub and Spoke" modular architecture managed by a central `ActiveLearningOrchestrator`. It completely separates physics engines (LAMMPS/Quantum ESPRESSO) from the machine learning infrastructure (Pacemaker) and ensures safe inter-process communication using strict Data Transfer Objects.

```mermaid
graph TD
    User([User Config]) --> ConfigModel[Strict Pydantic Config]
    ConfigModel --> Orchestrator

    subgraph Active Learning Loop
        Orchestrator --> |1. Run Exploration| DynamicsEngine[Dynamics Engine\nMD/kMC]
        DynamicsEngine --> |2. HaltInfo \n(Uncertainty Detected)| Orchestrator
        Orchestrator --> |3. Generate Candidates| StructureGenerator[Structure Generator]
        StructureGenerator --> |4. Perturbed Structures| Orchestrator
        Orchestrator --> |5. Compute Ground Truth| Oracle[DFT Oracle]
        Oracle --> |6. Evaluated Structures| Orchestrator
        Orchestrator --> |7. Update Dataset & Train| Trainer[ACE Trainer]
        Trainer --> |8. New Potential| Orchestrator
        Orchestrator --> |9. Validate Potential| Validator[Quality Validator]
        Validator --> |10. Validation Result| Orchestrator
    end

    Orchestrator --> |11. Deploy Potential| FileSystem[(Persistent Storage)]
    FileSystem --> |12. Resume Run| DynamicsEngine
```

## Prerequisites

*   **Python:** 3.12+
*   **Package Manager:** `uv`
*   **Engines (Real Mode):** LAMMPS (`lmp`), EON (`eonclient`), Quantum ESPRESSO, and Pacemaker (`pace_train`).
    *(Note: You can run the entire pipeline in Mock Mode for development without these heavy dependencies).*

## Installation & Setup

We recommend using `uv` for fast dependency management and virtual environment creation.

```bash
# Clone the repository
git clone https://github.com/your-org/mlip-pipelines.git
cd mlip-pipelines

# Sync dependencies and create a virtual environment
uv sync

# Configure the environment
cp .env.example .env

# Verify the installation by running the test suite
uv run pytest
```

## Usage

### Quick Start

Ensure your `.env` is configured correctly, then run the interactive Marimo tutorial. The tutorial encompasses a complete pipeline run in an isolated mock environment, showcasing the OTF mechanism.

```bash
uv run marimo run tutorials/UAT_AND_TUTORIAL.py
```

For real system execution, you will generally define your system in a `config.yaml` and invoke the orchestrator via the CLI (implementation specific to your deployment strategy).

## Development Workflow

The project is developed in 6 discrete cycles. All components enforce strict Typing and Linter standards to maintain high AI code generation quality.

*   **Running Linters:**
    ```bash
    uv run ruff check
    uv run mypy src tests
    ```

*   **Running Tests:**
    Tests strictly avoid polluting the global filesystem using `tmp_path` fixtures.
    ```bash
    uv run pytest --cov=src --cov-report=term-missing
    ```

## Project Structure

```text
mlip-pipelines/
├── src/
│   ├── core/           # Interfaces and the central Orchestrator
│   ├── domain_models/  # Pydantic Schemas and DTOs
│   ├── dynamics/       # LAMMPS and EON interfaces
│   ├── generators/     # Structure and Defect generation
│   ├── oracles/        # Quantum ESPRESSO DFT calculations
│   ├── trainers/       # ACE Pacemaker wrappers
│   └── validators/     # Quality assessment and stability checks
├── tests/              # Comprehensive test suite (unit, integration, e2e)
├── tutorials/          # Executable Marimo tutorials for UAT
└── pyproject.toml      # Project configuration
```

## License

This project is licensed under the MIT License.
