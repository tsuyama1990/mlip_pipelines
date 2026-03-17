# Adaptive-MLIP Platform

## Overview
Adaptive-MLIP is a next-generation automated framework that intelligently links Density Functional Theory (DFT), Machine Learning Interatomic Potentials (MLIP), and Molecular Dynamics/Monte Carlo simulations. It is designed to empower researchers by providing a complete Intent-Driven user experience, completely abstracting away the complex parameters typically required to run sophisticated active-learning workflows for materials science.

## Features
- **Intent-Driven Configuration API**: Provide high-level intent, like target materials and an Accuracy vs. Speed tradeoff slider, and the framework automatically maps it to precise numerical physics hyper-parameters.
- **Strict Security Guarantees**: Robust configuration validation, ensuring absolute paths, file-sizes, symlinks, and user inputs are strictly bounded and safe to run on HPC clusters.
- **RESTful API Gateway**: Seamlessly integrates with modern web UIs (such as React Flow) via FastAPI, translating visually designed execution workflows into deeply nested YAML constraints in real time.

## Installation
The system uses `uv` for modern, blazingly fast dependency management.
```bash
# Clone the repository and sync dependencies
git clone https://github.com/your-org/adaptive-mlip.git
cd adaptive-mlip

# Install all runtime dependencies and setup the virtual environment
uv sync
```

## Usage
The core engine can be configured via an HTTP REST endpoint designed to receive intent-driven configurations from the GUI.

**Starting the Backend Server:**
```bash
# Run the FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Submitting a Configuration Payload via Python:**
```python
import httpx

payload = {
    # Provide necessary fields: project_root, system, dynamics, oracle, etc.
    # ...
    "intent": {
        "target_material": "Pt-Ni",
        "accuracy_speed_tradeoff": 7, # 1 (Fast) to 10 (Most Accurate)
        "enable_auto_hpo": True
    }
}

response = httpx.post("http://localhost:8000/config/submit", json=payload)
print(response.json())
```

## Structure
- `src/domain_models/`: Pydantic configurations defining strict system boundaries and user intents.
- `src/api/`: FastAPI gateway handling GUI to orchestration translations.
- `src/dynamics/`: Safe security utilities and environment validators.
- `tests/`: Extensive Test-Driven Development (TDD) coverage ensuring zero-mock physical validation and deterministic state resolution.
