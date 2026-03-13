# MLIP Pipelines

An automated, zero-configuration pipeline for building, validating, and deploying machine learning interatomic potentials (MLIPs).

It automatically explores structural space, labels structures via DFT, trains ACE potentials iteratively, and dynamically resumes Molecular Dynamics execution loops using a robust internal state machine.

## Overview

The `mlip-pipelines` system automates the traditionally manual process of generating highly accurate Atomic Cluster Expansion (ACE) potentials. By leveraging "On-The-Fly" (OTF) uncertainty monitoring, the system performs targeted molecular dynamics (MD) exploration and runs DFT labeling selectively on configurations that exceed known uncertainty thresholds.

## Features

- **Adaptive Exploration**: Dynamically decides exploration policies (Defect-Driven, Strain-Heavy, High-MC) based on material DNA.
- **Self-Healing DFT Oracle**: Automates Quantum ESPRESSO via `ase` with auto-adjusting parameters (e.g. `mixing_beta`) if SCF convergence fails.
- **Smart Sampling & D-Optimality**: Identifies high-gamma structures in LAMMPS and downselects actively using `pace_activeset` and localized generation.
- **Automated Validation**: Integrated `phonopy` stability and stress-strain calculations for elastic moduli verification.
- **Zero-Config Execution**: Entire multi-stage generation from exploration to deployment executes autonomously from a single `config.yaml` or `.env` configuration.

## Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

## Quick Start / UAT Tutorial

The project comes with a built-in User Acceptance Test (UAT) implemented as an interactive `marimo` notebook. This demonstrates the cycle of Halt & Heal and Interface computation for an FePt/MgO system.

```bash
# To run the tutorial headlessly (testing logic)
uv run python tutorials/uat_and_tutorial.py

# To open the interactive notebook
uv run marimo edit tutorials/uat_and_tutorial.py
```

## Structure

- `src/core/orchestrator.py`: Orchestrates the Active Learning cycle.
- `src/domain_models/`: Pure Pydantic data schemas defining configurations.
- `src/dynamics/`: LAMMPS MD integration with uncertainty monitors.
- `src/oracles/`: Quantum ESPRESSO integration with automatic parameter fallback.
- `src/trainers/`: Pacemaker ACE configuration.
- `src/validators/`: Phonon & mechanical stability assurance.
