# Adaptive-MLIP GUI Platform

## Overview
The Adaptive-MLIP GUI Platform is a modern, intent-driven interface for constructing and managing advanced Machine Learning Interatomic Potential (MLIP) pipelines. It translates high-level research objectives into robust, heavily validated computational workflows.

## Features
- **Intent-Driven Configuration**: A simple Accuracy vs. Speed slider automatically determines the optimal hyperparameter bounds, eliminating the need to manually configure thresholds for Active Learning.
- **Strict Security**: The underlying API layer continuously validates all inputs, ensuring no malicious paths or payload bloats can compromise your compute environment.
- **RESTful Translation Gateway**: Safely manages state between frontend GUI models and deep backend Pydantic models.

## Installation
Ensure you have `uv` installed, then synchronize the environment:
```bash
uv sync
```

## Usage
Start the backend FastAPI server:
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
You can access the API documentation at `http://localhost:8000/docs`.

## Structure
- `src/api/`: FastAPI server and route configurations.
- `src/domain_models/`: Robust, strictly typed Pydantic data schemas defining system state.
