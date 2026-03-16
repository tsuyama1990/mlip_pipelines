# MLIP NextGen Orchestration

Welcome to the MLIP NextGen Orchestration pipeline. This tool provides an automated, fault-tolerant active learning framework for discovering and refining Machine Learning Interatomic Potentials (MLIPs).

## Features

- **Continuous Active Learning Loop:** Completely automated 4-phase hierarchical distillation (Distillation, Validation, Exploration, Extraction/Finetuning).
- **HPC Fault Tolerance & Checkpointing:** Built-in SQLite state management ensures that if jobs are abruptly killed by Slurm wall-time limits, the workflow instantaneously resumes without repeating heavy DFT evaluations.
- **Automated Artifact Cleanup:** Features an aggressive, asynchronous cleanup daemon that compresses or deletes massive temporary files (`.wfc`, LAMMPS `.dump`) to strictly respect HPC storage quota limits.
- **Tiered Oracles:** Intelligently switches between fast empirical models and heavy Density Functional Theory (Quantum Espresso) strictly based on uncertainty thresholds.
- **Adaptive Exploration:** Uses local active set selection and adaptive policy engines to maximize simulation space exploration.

## Installation

Ensure you have Python 3.12+ installed, and then use `uv` for lightning-fast dependency synchronization:

```bash
# Clone the repository
git clone https://github.com/organization/mlip-nextgen.git
cd mlip-nextgen

# Sync dependencies using uv
uv sync
```

## Usage

Configuration relies on standard `.env` and `ProjectConfig` Pydantic models. To run the full orchestration pipeline programmatically:

```python
from pathlib import Path
from src.core.orchestrator import Orchestrator
from src.domain_models.config import ProjectConfig

# Load configuration and establish root
config = ProjectConfig(project_root=Path("/path/to/project"))

# Initialize the automated pipeline
orchestrator = Orchestrator(config)

# Run a single active learning cycle (this can be placed in a while loop)
result = orchestrator.run_cycle()
print(f"Cycle finished with potential: {result}")
```

## Structure Overview

- `src/core/orchestrator.py`: The main infinite loop and brain of the active learning pipeline.
- `src/core/checkpoint.py`: SQLite-backed state manager protecting against cluster crash interruptions.
- `src/dynamics/`: Modules for driving LAMMPS MD and EON kMC exploration.
- `src/oracles/`: Evaluators including fast ACE/MACE managers and precise DFT runners.
- `src/trainers/`: Modules handling data aggregation and finetuning of potentials.
