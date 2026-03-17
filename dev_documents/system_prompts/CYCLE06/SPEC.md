# CYCLE06: Dynamic Tasks, Final Refactoring, & Stabilization Specification

## 1. Summary

CYCLE06 constitutes the final, critical architectural sprint for the Adaptive-MLIP GUI backend. Its primary objective is twofold, demanding absolute rigor. First, it completely implements the complex "Dynamic LAMMPS Tasks" requirement specified in the PRD, specifically focusing on the intent-based, GUI-driven setup for advanced Grand Canonical Monte Carlo (GCMC) gas adsorption and molecular film deposition workflows. In traditional systems, users must manually calculate and input highly complex thermodynamic potentials. In this advanced cycle, users will seamlessly input only high-level thermodynamic variables—Target Gas Species (e.g., O2), Target Temperature (K), and Target Pressure (bar)—and the intelligent backend will automatically compute the exceedingly complex chemical potentials ($\mu$) using rigorous statistical mechanics, and perfectly compile the dense, argument-heavy `fix gcmc` command blocks.

Second, this entire cycle is dedicated to comprehensive, project-wide system stabilization, uncompromising code quality enforcement, and extensive User Acceptance Testing (UAT) preparation. We will execute rigorous, uncompromising linting utilizing `ruff` (enforcing strict McCabe complexity limits across every single function), execute absolute type checking using `mypy --strict`, and beautifully consolidate all UAT scenarios from cycles 01 through 06 into a singular, highly executable, interactive Marimo notebook (`tutorials/adaptive_mlip_gui_workflow.py`). This guarantees the API and the massive configuration engine function perfectly, seamlessly, and securely end-to-end, ready for production deployment on enterprise HPC clusters.

## 2. System Architecture

This cycle significantly focuses on expanding the sophisticated capability of `lammps_generator.py` and deeply extending the `DynamicsConfig` Pydantic models, followed by an aggressive, project-wide architectural refactoring to remove any accumulated technical debt.

**File Structure (ASCII Tree):**
```text
mlip-pipelines/
├── src/
│   ├── domain_models/
│   │   ├── config.py                   # Extended deeply with DynamicTaskConfig validations
│   │   └── gui_schemas.py              # Extended with Thermodynamic intent models
│   ├── dynamics/
│   │   ├── **thermo_solver.py**            # New rigorous utility for calculating precise chemical potentials from equations of state
│   │   └── lammps_generator.py         # Heavily extended to securely compile complex `fix gcmc` and `fix deposit` commands
├── tutorials/
│   └── **adaptive_mlip_gui_workflow.py**   # The final, master UAT/Tutorial interactive Marimo notebook
└── pyproject.toml                      # Final linter verification and strict dependency lockdown
```

The system architecture data flow operates as follows: When a user configures a complex dynamic task in the React GUI, they provide simple inputs like `target_pressure` and `target_temperature`. The incredibly strict `DynamicsConfig` validator intercepts these raw numbers. It instantly calls the newly constructed `thermo_solver.py` module, which meticulously utilizes standard Ideal Gas equations, rigorous statistical mechanics, or customized empirical equations of state to compute the exact chemical potential $\mu$. This calculated $\mu$ is safely, immutably stored within the validated Pydantic model. During execution, `lammps_generator.py` securely extracts $\mu$ and writes the intricate, multi-line `fix gcmc` or `fix deposit` script blocks, automatically and securely handling random number seeds, molecule template generation, and boundary constraints without user intervention.

## 3. Design Architecture

**Domain Concepts & Pydantic Models:**
-   **`DynamicTaskConfig`**: A brand new, highly complex model securely added to the core `DynamicsConfig`.
    -   `task_type`: A strict Enum natively limited to (`MD`, `GCMC`, `DEPOSITION`).
    -   `gas_species`: A sanitized String representing the molecule (e.g., "O2", "H2").
    -   `target_temperature`: A heavily validated Float (must be strictly > 0 Kelvin).
    -   `target_pressure`: A heavily validated Float (must be strictly > 0, typically in atm or bar).
    -   `chemical_potential`: A strictly derived Float (in eV, automatically calculated exclusively by the internal `@model_validator`, never accepted directly from the frontend payload).

**Key Invariants, Constraints, and Validation Rules:**
1.  **Absolute Thermodynamic Validity Enforcement**: The system must enforce physical reality. `target_temperature` must be strictly positive (> 0.0 K), as absolute zero or negative temperatures will catastrophically crash the thermodynamic solver's logarithmic functions. `target_pressure` must be strictly positive. Any violation immediately returns an HTTP 422.
2.  **Immutable Chemical Potential Calculation**: The highly complex `chemical_potential` field is strictly read-only from the user's perspective. It must be computed dynamically inside the deeply nested `ProjectConfig` validation flow using the provided T and P. If a malicious or misconfigured GUI client attempts to manually override it in the incoming JSON payload, the Pydantic validator must aggressively log a warning and silently, forcefully recompute it from scratch to ensure absolute physical correctness and prevent simulation sabotage.
3.  **Uncompromising Linting and Typing**: The entire Python codebase must absolutely pass all `ruff` complexity checks (`mccabe < 10` limits) and run entirely clean under `mypy --strict`. Any duplicated validation logic discovered during earlier cycles must be ruthlessly extracted and cleanly centralized into `src/dynamics/security_utils.py` (or a similarly purposed secure utility module) to eliminate technical debt entirely.

## 4. Implementation Approach

**Step 1: Thermodynamic Solver Implementation and Statistical Mechanics**
Carefully construct the completely new `src/dynamics/thermo_solver.py` module. Implement a highly robust function `calculate_chemical_potential(species: str, T: float, P: float) -> float`. For the scope of this initial GUI translation cycle, a simplified Ideal Gas approximation utilizing standard entropy/enthalpy lookup tables (or an integrated ASE `Thermochemistry` module if deeply available) is sufficient to brilliantly demonstrate the architectural pipeline, provided the math handles extreme edge cases gracefully without throwing unhandled exceptions.

**Step 2: Pydantic Schema Extension and Data Immutability**
Define the `DynamicTaskConfig` inside the core `config.py` module. Expertly implement the critical `@model_validator(mode="after")` function that elegantly extracts `self.target_temperature` and `self.target_pressure`, safely passes them to the aforementioned solver, catches any physical boundary errors, and conclusively assigns the precise result strictly to the `self.chemical_potential` attribute, completely discarding any user-provided garbage data in that field.

**Step 3: Extending the LAMMPS Generator for Advanced Topologies**
Significantly modify the `lammps_generator.py` compiler. If the `task_type == GCMC` Enum is detected, entirely suppress the standard NVE/NVT integration blocks. Instead, dynamically and flawlessly construct the notoriously difficult `fix gcmc` command string.
Example formulation: `fix 1 active_group gcmc 100 100 100 1 29815 {target_temperature} {chemical_potential} 0.5`
Ensure that accompanying molecule templates (if required by the specific gas species) are securely generated and accurately injected into the script, handling random number seeds securely using Python's `secrets` module instead of predictable `random` calls to ensure perfect reproducibility and security.

**Step 4: Comprehensive UAT Marimo Notebook Creation and Orchestration**
Create the masterpiece `tutorials/adaptive_mlip_gui_workflow.py`. This single, beautifully documented file will programmatically, effortlessly walk through every single UAT scenario from Cycle 01 through Cycle 06. It will utilize the standard Python `requests` library (or FastAPI TestClient directly for speed) to perfectly simulate the React frontend posting complex JSON payloads to the API. It will actively capture and beautifully display the JSON responses, decisively demonstrate the Run 0 instantaneous validation, and securely start and pause an Orchestrator run, serving as the ultimate integration test and user tutorial.

## 5. Test Strategy

Testing this massive final cycle focuses intently on absolute thermodynamic correctness, flawless LAMMPS script generation accuracy, and full, uncompromising end-to-end integration via the interactive Marimo notebook environment.

**Unit Testing Approach:**
-   **Target:** The `thermo_solver.py` module and mathematical boundaries.
-   **Method:** We will strongly assert that calling `calculate_chemical_potential("O2", 300, 1.0)` reliably returns a physically reasonable negative float value (e.g., matching standard CRC tables within a 5% margin). We will aggressively assert that passing the impossible T=0 K or a negative pressure raises a brilliant, highly descriptive `ValueError` instantly.
-   **Target:** `DynamicTaskConfig` strict Pydantic validation.
-   **Method:** We will forcefully instantiate the config with valid T=300, P=1, while deliberately injecting a completely wrong, dummy value into the `chemical_potential` field. We will confidently assert that the resulting `chemical_potential` attribute is populated correctly by the solver, completely and silently ignoring the dummy value passed in the initial constructor dictionary, proving data immutability.

**Integration Testing Approach:**
-   **Target:** The massive `lammps_generator.py` GCMC output compiler.
-   **Method:** Generate a complete LAMMPS script populated with a valid `DynamicTaskConfig` explicitly for GCMC. We will use incredibly precise regex matching to assert that the exact string `fix gcmc` is present, flawlessly formatted, and that the calculated `chemical_potential` string exactly matches the expected float value in the perfectly correct argument position, avoiding off-by-one string formatting errors.
-   **Target:** The E2E Pipeline and Marimo execution.
-   **Method:** Execute the `tutorials/adaptive_mlip_gui_workflow.py` Marimo notebook completely programmatically (`uv run python tutorials/adaptive_mlip_gui_workflow.py`) within the automated CI pipeline.
-   **Side-effect Isolation:** The interactive Marimo notebook will brilliantly insert `sys.path.insert(0, os.getcwd())` at the absolute top of its context to natively resolve local modules correctly without ever breaking or polluting the global Python environment. It will aggressively utilize a `tempfile.TemporaryDirectory` for absolutely all simulated API database checkpointing and massive file I/O operations, ensuring zero artifacts (not a single byte) are left on the developer's machine upon a successful exit with code 0.
