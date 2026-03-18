# PyAcemaker

## Overview
PyAcemaker is a comprehensive automation pipeline for generating highly accurate Machine Learning Interatomic Potentials (MLIPs). It orchestrates advanced Molecular Dynamics (MD) and Adaptive Kinetic Monte Carlo (aKMC) simulations, automates Quantum Espresso (DFT) calculations with self-healing routines, and fine-tunes base models like MACE with active learning protocols using Atomic Cluster Expansion (ACE). The system is highly resilient and guarantees physical accuracy of the resulting potentials by ensuring robust Lennard-Jones baselines, rigorous quality assurance, and automated surface passivation.

## Features
- **Highly Configurable**: Control complex multi-stage MLIP generation with a single valid YAML specification.
- **Resilient Orchestration**: State management with SQLite-based checkpointing ensuring long-running processes survive job kills.
- **Automated QA & Physical Validation**: Automatic detection of imaginary phonon frequencies and elastic tensor checks using built-in validators with `phonopy`.
- **Intelligent Fault Tolerance**: Automated "cut-out" generation for regions exceeding expected extrapolation uncertainties.
- **Seamless Integrations**: Tight coupling with robust computational engines like LAMMPS, Quantum Espresso, and MACE base models.

## Installation
Ensure you have `uv` installed, then synchronize the environment:
```bash
uv sync
```

## Usage
Run the main PyAcemaker orchestration flow (ensure your config `.env` and `input.yaml` are correctly set):
```python
from src.core.orchestrator import Orchestrator
from src.domain_models.config import ProjectConfig

config = ProjectConfig()
orchestrator = Orchestrator(config)
orchestrator.run_cycle()
```

## Project Structure
- `src/domain_models/`: Pydantic models acting as strict type-safe data contracts.
- `src/core/`: Orchestration and execution pipelines.
- `src/dynamics/`: MD and aKMC wrappers for simulators like LAMMPS and EON.
- `src/trainers/`: Training routines, Finetune managers, and incremental updates.
- `src/validators/`: Stability and quality assurance tools.
