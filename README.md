# PyAcemaker: MLIP Pipelines

An automated system for building and operating state-of-the-art Machine Learning Interatomic Potentials (MLIPs).

## Overview
PyAcemaker simplifies the process of creating high-quality MLIPs, lowering the barrier to entry for researchers by automating complex workflows like structure generation, DFT calculation, potential training, and validation.

## Features
- **Zero-Config Workflow**: Drive the entire computational pipeline using a single configuration file.
- **Data Efficiency**: Uses active learning strategies to minimize the need for expensive DFT calculations.
- **Physics-Informed Robustness**: Ensures physical safety in extrapolation regions via baseline delta-learning techniques (e.g., Lennard-Jones/ZBL).
- **Seamless Resume**: Recovers from uncertainty events and continues molecular dynamics simulations automatically.
- **Hierarchical Distillation**: Transfers generalization capabilities from foundational models like MACE-MP-0 to ACE frameworks.

## Installation & Setup
Requires Python 3.12+ and `uv`.
```bash
git clone https://github.com/example/mlip-pipelines.git
cd mlip-pipelines
uv sync
```

## Usage
Execute the main orchestrator script to initiate the simulation workflow:
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
