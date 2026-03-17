# Adaptive-MLIP GUI Platform

## Overview
The Adaptive-MLIP GUI Platform is an advanced, fully automated Machine Learning Interatomic Potential (MLIP) construction and operational pipeline. It seamlessly integrates first-principles calculations (DFT) with Machine Learning Potentials (MLIPs like MACE and ACE) into a cohesive intent-driven system.

By bridging the gap between materials science objectives and the intricate syntax of high-performance computing tools, this platform ensures that experimentalists and computational researchers can focus on the physics of their problem domain rather than complex configurations. It translates high-level intents—such as a simple "Accuracy vs. Speed" slider—into mathematically precise, robust hyperparameter configurations required to orchestrate complex molecular simulations without error.

## Features
- **Intent-Driven Configuration**: Configure complex Active Learning environments by simply defining a target material and an Accuracy vs. Speed tradeoff via a unified FastAPI gateway.
- **Strict Boundary Security Validation**: Prevents path traversal sequences, shell injections, and malformed configurations by employing highly secure Pydantic model validators.
- **RESTful API Endpoint**: Easily integrate your intent-driven UI payloads using our `/config/submit` endpoint which guarantees mathematical and structural validity before computation.
- **Automated Mathematical Tradeoffs**: Dynamically determines uncertainty thresholds, replay buffer sizes, and training epochs to safely manage model knowledge retention vs. exploration speed.

## Installation

Ensure you have Python >= 3.12 and [uv](https://github.com/astral-sh/uv) installed, then run:

```bash
uv sync
```

## Usage

To start the local FastAPI web server which processes GUI configurations:

```bash
uv run uvicorn src.api.main:app --reload
```

You can then test the API endpoint using a simple `curl` command:

```bash
curl -X POST "http://127.0.0.1:8000/config/submit" \
     -H "Content-Type: application/json" \
     -d '{
       "project_root": "/tmp/my_mlip_project",
       "system": {"elements": ["Fe"]},
       "dynamics": {"trusted_directories": [], "project_root": "/tmp/my_mlip_project"},
       "oracle": {},
       "trainer": {"trusted_directories": []},
       "validator": {},
       "distillation_config": {"temp_dir": "/tmp", "output_dir": "/tmp", "model_storage_path": "/tmp"},
       "loop_strategy": {"replay_buffer_size": 500, "checkpoint_interval": 5, "timeout_seconds": 86400},
       "intent": {
         "target_material": "Fe",
         "accuracy_speed_tradeoff": 8,
         "enable_auto_hpo": false
       }
     }'
```

## Structure
- `src/api/`: Contains the FastAPI web application, API routes, and CORS middleware logic.
- `src/domain_models/`: Contains the strict Pydantic schemas validating both core backend parameters and GUI specific translation payloads (`gui_schemas.py`).
- `src/dynamics/`: Houses the dynamics execution logic and robust string sanitization security mechanisms.
- `src/core/`: Contains the main Active Learning orchestration algorithms.
