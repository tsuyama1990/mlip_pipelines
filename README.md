# Adaptive-MLIP: Zero-Config Active Learning

## Overview
Adaptive-MLIP is an intelligent, automated pipeline designed to democratise the construction and deployment of State-of-the-Art Machine Learning Interatomic Potentials (MLIP). By drastically simplifying the hyperparameter configuration process through an "Intent-Driven" Graphical User Interface architecture, this platform allows computational chemists and materials scientists to focus entirely on their domain problems rather than managing complex simulation orchestrations.

## Features
*   **Intent-Driven Configuration**: A robust FastAPI gateway translates high-level user intents (e.g., "Accuracy vs. Speed" tradeoffs) into precise, physically valid hyperparameters, eliminating the need to write complex YAML scripts.
*   **Secure API Boundary**: The REST API layer strictly sanitizes incoming data against path traversals, injection attacks, and memory bloat, completely isolating the sensitive HPC orchestrator from malformed web inputs.
*   **Zero-Configuration Execution**: Deeply nested orchestration schemas are automatically generated, synced, and validated using advanced Pydantic `model_validators`, enforcing physical invariants and preventing catastrophic out-of-bounds execution errors.

## Installation
The project uses `uv` for fast, reproducible dependency management. To set up your local environment:

```bash
# Clone the repository
git clone <repository_url>
cd mlip-pipelines

# Synchronize all dependencies and create the virtual environment
uv sync
```

## Usage
The primary interaction with the system is through the REST API gateway, which receives configuration intents and validates them before passing them to the Orchestrator.

To start the FastAPI backend server:

```bash
uv run uvicorn src:app --host 0.0.0.0 --port 8000
```

You can then submit your simulation intent via HTTP POST. The system will automatically translate your `accuracy_speed_tradeoff` into the deeply nested active learning thresholds.

```bash
curl -X POST http://localhost:8000/config/submit \
  -H "Content-Type: application/json" \
  -d '{
    "project_root": "/absolute/path/to/project",
    "system": { "elements": ["Fe", "Pt"] },
    "dynamics": {
      "project_root": "/absolute/path/to/project",
      "trusted_directories": ["/usr/bin"]
    },
    "oracle": {},
    "trainer": { "trusted_directories": ["/usr/bin"] },
    "validator": {},
    "distillation_config": {
      "temp_dir": "/tmp",
      "output_dir": "/tmp",
      "model_storage_path": "/tmp"
    },
    "loop_strategy": {
      "replay_buffer_size": 500,
      "checkpoint_interval": 5,
      "timeout_seconds": 3600
    },
    "intent": {
      "target_material": "FePt",
      "accuracy_speed_tradeoff": 5,
      "enable_auto_hpo": true
    }
  }'
```

## Structure
*   `src/`: Contains the core application logic.
    *   `src/__init__.py`: The FastAPI application entry point.
    *   `src/domain_models/`: The Pydantic Data Transfer Objects (`dtos.py`) and central configuration definitions (`config.py`).
    *   `src/core/`: The Active Learning state machine and Orchestrator.
    *   `src/dynamics/`, `src/generators/`, `src/oracles/`, `src/trainers/`, `src/validators/`: Modular simulation execution components.
*   `tests/`: Comprehensive test suite including unit, integration (e2e), and User Acceptance Testing (`uat/`) executable Marimo notebooks.
