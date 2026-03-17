# CYCLE06: Dynamic Tasks, Final Refactoring, & Stabilization UAT Plan

## 1. Test Scenarios

### Scenario ID: UAT-06-01: Thermodynamic Translation for Dynamic Tasks
**Priority:** High
**Description:** Verify that the system correctly, instantly, and flawlessly translates high-level thermodynamic intents (e.g., Target Temperature and Target Pressure specifically for a given gas species like Oxygen or Hydrogen) into the exact, mathematically correct chemical potential ($\mu$) required natively by the LAMMPS C++ engine. The user seamlessly submits only T and P via the intuitive GUI; the backend must precisely calculate $\mu$ and flawlessly compile the incredibly dense, error-prone, 8+ argument `fix gcmc` command block automatically, totally shielding the user from statistical mechanics calculations.

### Scenario ID: UAT-06-02: Strict Enforcement of Thermodynamic Invariants
**Priority:** High
**Description:** Verify that the robust backend strictly and aggressively rejects any incoming API payload attempting to set completely unphysical thermodynamic properties, such as a negative or zero absolute temperature (which is physically impossible), or a negative pressure. This critical security validation strictly prevents the underlying thermodynamic solver from throwing unhandled internal mathematical errors (e.g., attempting to calculate the logarithms of negative numbers) and consequently crashing the entire FastAPI application layer.

### Scenario ID: UAT-06-03: End-to-End Marimo Tutorial Execution and Validation
**Priority:** Critical
**Description:** Verify that the comprehensively consolidated Marimo tutorial notebook (`tutorials/adaptive_mlip_gui_workflow.py`) executes perfectly, entirely without errors from top to bottom. This interactive notebook acts as the master end-to-end integration test for the entire GUI platform project, programmatically and seamlessly walking through the complete API lifecycle: connecting securely via WebSocket, successfully submitting a Run 0 pre-flight validation payload, perfectly configuring an asynchronous Auto-HPO task, flawlessly translating an intent-driven GUI payload into a deeply nested `ProjectConfig`, and finally securely launching, pausing, and resuming the massive HPC Orchestrator. If this notebook passes, the architecture is fundamentally sound.

## 2. Behavior Definitions

### UAT-06-01: Thermodynamic Translation for Dynamic Tasks
**GIVEN** a running, healthy instance of the Adaptive-MLIP FastAPI backend
**AND** a simulated GUI JSON payload meticulously containing a `DynamicTaskConfig` requesting `task_type: GCMC`, `gas_species: O2`, `target_temperature: 300.0` (in Kelvin), and `target_pressure: 1.0` (in bar)
**WHEN** the extensive payload is cleanly submitted to the API and validated by the Pydantic engine
**THEN** the returned serialized JSON configuration clearly shows that the highly complex `chemical_potential` was automatically calculated and correctly populated (e.g., ~-0.5 eV)
**AND** the securely generated LAMMPS script firmly contains the exact `fix gcmc` command utilizing that precise calculated chemical potential value in the proper argument position without syntax errors.

### UAT-06-02: Strict Enforcement of Thermodynamic Invariants
**GIVEN** a running instance of the Adaptive-MLIP FastAPI backend fully exposed to the network
**AND** a simulated, malformed GUI JSON payload erroneously or maliciously containing a `DynamicTaskConfig` with the impossible `target_temperature: 0.0` or `target_pressure: -1.0`
**WHEN** the flawed payload is submitted via HTTP POST to the `/config/submit` or `/validate/run0` endpoint
**THEN** the system instantly, categorically rejects the payload before processing
**AND** responds quickly with an HTTP 422 Unprocessable Entity status code
**AND** the JSON error details explicitly, clearly indicate that Temperature must be strictly greater than 0 K and Pressure must be positive to obey the laws of thermodynamics
**AND** the complex thermodynamic solver is absolutely never invoked, saving compute cycles.

### UAT-06-03: End-to-End Marimo Tutorial Execution and Validation
**GIVEN** a completely fresh, strictly typed Python 3.12 environment with all `uv` dependencies flawlessly installed
**WHEN** the terminal command `uv run python tutorials/adaptive_mlip_gui_workflow.py` is executed locally or on the CI runner
**THEN** the script successfully initializes a clean, isolated temporary test directory
**AND** sequentially, perfectly simulates absolutely all API calls defined in the architecture (Run 0, HPO, Config Translation, Orchestrator Start/Pause)
**AND** every single rigorous assertion in the interactive notebook passes successfully without raising any exceptions
**AND** the massive script gracefully exits with a return code of 0, leaving absolutely no residual HPC artifacts, databases, or massive trajectory files on the host file system.
