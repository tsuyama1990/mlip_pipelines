# MLIP Pipelines

A robust automated machine learning interatomic potential (MLIP) generator and validator framework.

## Overview

MLIP Pipelines orchestrates the complex workflow required to actively learn and deploy machine learning potentials. It autonomously explores configuration space using molecular dynamics, identifies regions of high uncertainty, queries an external DFT oracle for ground truth, trains updated potentials, and rigidly validates their physical accuracy and stability before deployment.

## Key Features

- **Active Learning Orchestration**: Fully automated loop from data generation to potential deployment.
- **Adaptive Exploration Policies**: Dynamically adjusts sampling strategies (MD vs MC, temperature scaling, defect introduction) based on material features.
- **Robust DFT Oracle Interface**: Integrates with standard atomistic simulation tools (VASP via ASE) to compute ground truth properties with built-in retry and failure handling.
- **Advanced Validator Phase **: Employs strict quality assurance gates. Tests deployed potentials against held-out datasets (Energy/Force RMSE) and validates physical stability criteria (Born stability, Phonon dispersion) using external packages (`phonopy`, ASE).
- **Automated HTML Reporting**: Generates comprehensive validation reports post-training to monitor potential quality and stability metrics.
- **Security & Sandboxing**: Strict validation of paths, configurations, and environment variables to prevent injection and traversal attacks.

## Installation

Ensure you have Python >= 3.12 installed, along with `uv` for dependency management.

```bash
uv sync
```

This will set up a virtual environment `.venv` with all necessary dependencies.

## Usage

To run the pipeline or interact with the modules programmatically:

```python
from pathlib import Path
from src.core.orchestrator import Orchestrator
from src.domain_models.config import ProjectConfig, SystemConfig

# 1. Define your project configuration
config = ProjectConfig(
    project_root=Path.cwd(),
    system=SystemConfig(elements=["Fe"]),
    # ... configure Dynamics, Oracle, Trainer, and Validator here ...
)

# 2. Initialize the Orchestrator
orchestrator = Orchestrator(config)

# 3. Run an Active Learning Cycle
result = orchestrator.run_cycle()

print(f"Cycle completed. Potential deployed at: {result}")
```

## Directory Structure

```text
mlip-pipelines/
├── src/
│   ├── core/           # Central Orchestrator
│   ├── domain_models/  # Pydantic schemas for data and config validation
│   ├── dynamics/       # Molecular dynamics and active sampling engines
│   ├── generators/     # Structure generation and adaptive policies
│   ├── oracles/        # Interface to DFT calculations
│   ├── trainers/       # ACE model training wrappers
│   └── validators/     # Quality assurance and reporting
├── tests/              # Comprehensive test suites (unit, e2e, uat)
└── pyproject.toml      # Project configuration and dependency definitions
```
