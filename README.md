# PyAcemaker: MLIP Pipelines

An automated system for building and operating state-of-the-art Machine Learning Interatomic Potentials (MLIPs).

## Overview
PyAcemaker simplifies the process of creating high-quality MLIPs, lowering the barrier to entry for researchers by automating complex workflows like structure generation, DFT calculation, potential training, and validation. The system is designed to autonomously handle complex physical failures, refining its mathematical understanding of interatomic potentials to build physically sound models without requiring human intervention.

## Features
- **Zero-Config Workflow**: Drive the entire computational pipeline using a single configuration file.
- **Physics-Informed Robustness**: Ensures physical safety in extrapolation regions via baseline delta-learning techniques (e.g., Lennard-Jones/ZBL).
- **Intelligent Error Handling**: Uses robust two-tier uncertainty thresholds to determine exactly when the system requires DFT quantum calculations versus generic machine learning predictions.
- **Automated Structure Passivation**: Intelligently extracts uncertain structures and passivates their boundaries to avoid artificial surface energy artifacts before executing rigorous DFT tasks.
- **Incremental Neural Updates**: Accelerates system training efficiently through dynamic replay buffers instead of relearning potentials from scratch.
- **Seamless Resume**: Recovers from uncertainty events and continues molecular dynamics simulations automatically from soft-start checkpoints.
- **Hierarchical Distillation**: Transfers generalization capabilities from foundational models like MACE-MP-0 to ACE frameworks to avoid expensive initial DFT calculations.

## Installation & Setup
Requires Python 3.12+ and `uv` package manager installed on your system.

```bash
git clone https://github.com/example/mlip-pipelines.git
cd mlip-pipelines
uv sync
```

## Usage
PyAcemaker is fully automated and designed to run persistently in the background. To initiate the simulation workflow with active learning:

```bash
uv run python src/core/orchestrator.py
```

## Project Structure
```text
.
├── src/               # Application source code
│   ├── core/          # Orchestrator and base classes
│   ├── domain_models/ # Pydantic configuration schemas
│   ├── dynamics/      # MD and kMC simulation engines
│   ├── generators/    # Structure and defect generation
│   ├── oracles/       # DFT and MACE calculation managers
│   ├── trainers/      # ACE potential training wrappers
│   └── validators/    # Physics and stability validation
├── tests/             # Unit, integration, and UAT tests
├── tutorials/         # Interactive Marimo notebooks
└── pyproject.toml     # Project dependencies
```

## License
MIT License
