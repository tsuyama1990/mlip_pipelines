# mlip-struc-gen-local

A local Python pipeline for crystal structure optimization using MACE (Machine Learning Interatomic Potentials). This project integrates structure generation logic from an external submodule with ASE-based optimization powered by MACE.

## Features

*   **Structure Generation**: Uses `mlip_struc_gen` (external submodule) to create initial structures (e.g., Alloys).
*   **Optimization**: Fully automated geometry optimization using `mace-torch` and `ase.optimize`.
*   **Robust Configuration**: Environment-based configuration using `pydantic-settings`.
*   **Logging**: structured logging via `loguru`.

## Directory Structure

```text
mlip-struc-gen-local/
├── .gitignore
├── .gitmodules
├── pyproject.toml
├── README.md
├── main_cli.py
├── src/
│   ├── config/              # Settings definition
│   ├── external/            # External repositories (git submodules)
│   ├── core/
│   │   ├── calculators/     # MACE factory
│   │   ├── engines/         # Relaxation engine
│   │   ├── generators/      # Adapter for external generator
│   │   └── utils/           # IO and logging
└── tests/                   # Unit and Integration tests
```

## Setup Instructions

### Prerequisites
*   Python 3.10+
*   [uv](https://github.com/astral-sh/uv) (Recommended for dependency management)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <this-repo-url>
    cd mlip-struc-gen-local
    ```

2.  **Initialize Submodules:**
    This project relies on an external submodule for structure generation.
    ```bash
    git submodule update --init --recursive
    ```

3.  **Install Dependencies:**
    Using `uv` to sync the environment (installs `torch`, `mace-torch`, `ase`, etc.):
    ```bash
    uv sync
    ```

    **Note on Torch/MACE Compatibility**:
    The project attempts to install `torch` and `mace-torch` compatible with your system. If you require specific CUDA versions (e.g., CUDA 11.8), ensure you install the correct PyTorch wheels beforehand or modify `pyproject.toml`.

## Usage

### 1. Configuration
The application is configured via environment variables or an `.env` file. A default run works out-of-the-box, but you can customize it:

**Example `.env` file:**
```ini
MACE__DEVICE=cuda           # Use 'cpu' if no GPU
MACE__MODEL_PATH=medium     # 'small', 'medium', 'large' or path to model file
RELAX__FMAX=0.01            # Convergence criterion (eV/A)
RELAX__STEPS=200            # Max steps
GENERATOR__TARGET_ELEMENT=Si # Element to generate (Alloy)
```

### 2. Run the Pipeline
Execute the main CLI entry point:

```bash
uv run main_cli.py
```

### 3. Output
Results are saved in `data/output/<timestamp>/`:
*   `final_structure.xyz`: Relaxed structure with energy/forces.
*   `trajectory.xyz`: Optimization trajectory.
*   `results.json`: Summary of the run (energy, convergence status).
*   `app.log`: Detailed logs.

## Testing

Run the test suite using `pytest`:

```bash
uv run pytest tests/
```
