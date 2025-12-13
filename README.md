# mlip-struc-gen-local

A local Python pipeline for crystal structure optimization using MACE (Machine Learning Interatomic Potentials). This project integrates structure generation logic from an external submodule with ASE-based optimization powered by MACE.

## Features

*   **Structure Generation**: Uses `mlip_struc_gen` (external submodule) to create initial structures (e.g., Alloys).
*   **Optimization**: Fully automated geometry optimization using `mace-torch` and `ase.optimize`.
*   **Physics Validation**: Automated checks for atomic clashes, cell validity, and physical constraints.
*   **Reproducibility**: Explicit random seed control for bit-identical structure generation.
*   **Robust Configuration**: Environment-based configuration using `pydantic-settings` with `.env` and `config.yaml` support.
*   **Logging**: Structured logging via `loguru`.

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
│   │   ├── validators/      # Physics checks
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
    This project relies on an external submodule for structure generation. **This step is critical.**
    ```bash
    git submodule update --init --recursive
    ```

3.  **Install Dependencies:**
    Using `uv` to sync the environment (installs `torch`, `mace-torch`, `ase`, etc.):
    ```bash
    uv sync
    ```

## Usage

### 1. Configuration
The application is configured via environment variables, an `.env` file, or `config.yaml`.
Priority: Environment Vars > `.env` > `config.yaml` > Defaults.

**Option A: `.env` file (Simple)**
```ini
MACE__DEVICE=cuda
GENERATOR__TARGET_ELEMENT=Si
RANDOM_SEED=12345
```

**Option B: `config.yaml` (Structured)**
```yaml
mace:
  device: cuda
  model_path: medium

generator:
  type: alloy
  target_element: Si
  supercell_size: 2

relax:
  fmax: 0.01

random_seed: 12345
```

### 2. Physics Validation
The pipeline enforces strict physical constraints:
*   **Atomic Clashes**: Rejects structures where atoms are closer than 60% of their combined covalent radii.
*   **Cell Validity**: Ensures simulation cells are non-singular and have positive volume.
*   **Periodic Boundaries**: Correctly handles distance calculations with Minimum Image Convention (MIC).

### 3. Reproducibility
Set `random_seed` in your configuration to ensure deterministic results.
*   The seed controls `numpy`, `torch`, and the structure generator.
*   Same seed + Same settings = Bit-identical structure.

### 4. Run the Pipeline
Execute the main CLI entry point:

```bash
uv run main_cli.py
```

### 5. Output
Results are saved in `data/output/<timestamp>/`:
*   `final_structure.xyz`: Relaxed structure with energy/forces.
*   `trajectory.xyz`: Optimization trajectory.
*   `results.json`: Summary of the run (energy, convergence status).
*   `app.log`: Detailed logs.

## Testing

Run the test suite using `pytest`. This includes unit tests, integration tests, and hypothesis-based property tests.

```bash
# Run all tests
uv run pytest tests/

# Run property-based tests only
uv run pytest tests/test_property_based.py
```
