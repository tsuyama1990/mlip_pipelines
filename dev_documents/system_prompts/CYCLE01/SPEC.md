# CYCLE01: Frontend Mockups & Core API Scaffolding Specification

## 1. Summary

The primary goal of CYCLE01 is to establish the foundational communication layer between the envisioned web-based Graphical User Interface (GUI) and the robust Python-based backend of the Adaptive-MLIP platform. This cycle will construct the essential FastAPI entry points and implement the new Pydantic Data Transfer Objects (DTOs) required to securely accept, parse, and validate the complex state objects emitted by the frontend. The Adaptive-MLIP framework relies on extremely precise hyperparameter settings to orchestrate the integration of Density Functional Theory, Foundation Models, and active learning dynamics. However, exposing these deeply nested parameters to end-users has proven to be a significant barrier to entry, resulting in frequent misconfigurations and catastrophic simulation failures.

Crucially, this cycle tackles the complex translation of the user's high-level intent into the specific, deeply nested configuration parameters that the existing `ProjectConfig` demands. The hallmark of this translation is the implementation of the "Accuracy vs. Speed" slider. Instead of overwhelming users with parameters like `candidate_threshold`, `sampling_structures_per_system`, or buffer radii, the GUI will send a single scalar value between 1 and 10. The backend must intercept this scalar and automatically calculate the mathematically optimal sub-parameters using rigorously defined functions. This abstraction is not merely cosmetic; it is a critical safety feature that guarantees the physical validity of the underlying simulation.

By the conclusion of this cycle, the system will possess a fully functional, highly secure REST API capable of receiving a serialized UI state, validating its structural integrity against common web-based injection attacks (such as path traversals or malicious string payloads), resolving mathematical trade-offs derived from user intent, and successfully instantiating a ready-to-execute `ProjectConfig` object in memory. This API layer serves as the absolute boundary between the untrusted web frontend and the highly sensitive, computationally expensive HPC core engine, ensuring that only perfectly formatted, physically sound configuration objects ever reach the orchestrator.

## 2. System Architecture

This cycle fundamentally modifies the boundary layer of the Adaptive-MLIP system. We will introduce a new top-level `src/api` module housing the FastAPI application logic and a new schema definition module `src/domain_models/gui_schemas.py`. Additionally, the existing `src/domain_models/config.py` will be safely extended to accept these translated intents via strict `@model_validator` functions, ensuring seamless backward compatibility with existing CLI execution workflows.

**File Structure (ASCII Tree):**
```text
mlip-pipelines/
├── src/
│   ├── **api**/
│   │   ├── **__init__.py**
│   │   ├── **main.py**                     # FastAPI application entry point and middleware config
│   │   └── **routes.py**                   # REST endpoints for config submission and validation
│   ├── core/
│   │   ├── orchestrator.py             # Existing Orchestrator (unmodified in this cycle)
│   ├── domain_models/
│   │   ├── __init__.py
│   │   ├── **config.py**                   # Existing schemas (extended with new validators)
│   │   └── **gui_schemas.py**              # New GUI-specific Pydantic DTOs for intent mapping
└── pyproject.toml
```

The data flow dictates that an HTTP POST request carrying a massive JSON payload arrives at `src/api/routes.py`. The payload is immediately serialized into the lightweight models defined in `gui_schemas.py`. Once the initial structure is verified (e.g., confirming strings do not contain path traversals (`../`) and that integers representing the UI sliders fall strictly within their defined bounds), the `@model_validator(mode="after")` methods integrated into `config.py` are triggered. These validation methods execute the mathematical translation of the 1-10 tradeoff slider into the actual `LoopStrategyConfig` and `DistillationConfig` numerical values required by the engine. If validation fails at any stage—either due to a type mismatch, a security violation, or a physically impossible derived state—FastAPI automatically intercepts the Pydantic `ValidationError` and returns a structured HTTP 422 error, completely shielding and protecting the core computational orchestrator from malformed inputs.

## 3. Design Architecture

The core of this cycle is the rigorous, uncompromising definition of the Pydantic schemas that represent the user's intent. Because this system drives massive HPC computations, the data modeling must be flawless, strictly enforcing data types, ranges, and cross-field relationships to prevent costly runtime errors.

**Domain Concepts & Pydantic Models:**
-   **`GUIStateConfig`**: Defined entirely within `gui_schemas.py`. This is a purely presentational, stateless schema designed to store the positions, zoom levels, and connections of the React Flow Directed Acyclic Graph (DAG) nodes present in the frontend. It contains absolutely zero physics logic. The backend simply receives it, validates its size to prevent memory bloat, and persists it to the database so the user can subsequently reload their visual workspace precisely as they left it.
-   **`WorkflowIntentConfig`**: Defined in `gui_schemas.py`. This crucial model captures the core scientific objective of the user. Key properties include:
    -   `target_material`: String (e.g., "Pt-Ni"). Must be strictly sanitized to prevent injection of malicious shell commands or path variables.
    -   `accuracy_speed_tradeoff`: Integer, strictly validated to be between 1 and 10 using Pydantic's `Field(ge=1, le=10)`. This is the primary driver of the backend's automatic parameter provisioning.
    -   `enable_auto_hpo`: Boolean. Flags whether the system should route into the Bayesian optimization paths (to be built in later cycles).
-   **`ProjectConfig` Extensions**: The existing `ProjectConfig` root model situated in `config.py` will be carefully modified to accept an optional `intent` attribute of type `Optional[WorkflowIntentConfig]`.

**Key Invariants, Constraints, and Validation Rules:**
1.  **Strict Security Boundaries**: All incoming strings (like `target_material` or generic descriptors) must be validated using centralized utility functions (e.g., rigorously checking for shell injection characters, directory traversals, or excessively long payloads that could trigger Denial of Service).
2.  **Trade-off Translation Mathematics**: The translation of the 1-10 slider must follow a predefined, monotonic mathematical scaling. For instance, if the slider is set to 1 (Maximum Speed), the resulting `uncertainty_threshold` in `DistillationConfig` might be aggressively high (e.g., 0.15), and `replay_buffer_size` might be minimal to conserve memory. If set to 10 (Maximum Accuracy), the threshold becomes extremely tight (e.g., 0.02), and the buffer size large to prevent forgetting. This translation logic is encapsulated exclusively within a `@model_validator(mode="after")` on the `ProjectConfig`.
3.  **Forward/Backward Compatibility**: The addition of the `WorkflowIntentConfig` must be strictly optional (`default=None`). Existing CLI-based test suites, automated CI pipelines, and expert users providing full, explicit YAML configurations must not be broken by the introduction of this new API layer. The validator will only override explicit configuration blocks if the `intent` object is actually provided and actively populated in the incoming HTTP payload.

## 4. Implementation Approach

The implementation will proceed in four distinct, logically isolated steps, focusing heavily on strict typing, data validation, and memory safety. We will leverage modern Python 3.12 features and strictly adhere to the project's Ruff formatting and McCabe complexity rules.

**Step 1: Scaffolding the Schema Extentions (`src/domain_models/gui_schemas.py`)**
We will create the new file dedicated entirely to UI-originated payloads. We will define the `GUIStateConfig` to hold arbitrary JSON-like dictionaries for React Flow data, employing Pydantic's `ConfigDict` to enforce strict size limits. Next, we will define the critical `WorkflowIntentConfig` with strict numeric bounds using Pydantic's `Field(ge=1, le=10)`. Ensure `extra="forbid"` is set on the `model_config` for all new models. This is a vital security requirement to prevent users from sending arbitrary payload bloat that could lead to Denial of Service (DoS) via memory exhaustion or unexpected dictionary unpacking errors deep within the application.

**Step 2: Extending `src/domain_models/config.py`**
We will modify the existing `ProjectConfig` model to smoothly incorporate the newly defined `WorkflowIntentConfig` as an optional field. The most complex engineering task here is implementing the critical `@model_validator(mode="after")` function. This function will first verify if `self.intent` exists. If it does, it will execute the mathematical mapping logic. For example, the derivation of the threshold could be implemented as: `calculated_threshold = 0.15 - (self.intent.accuracy_speed_tradeoff * 0.013)`. This rigorously derived floating-point value is then forcibly injected into `self.distillation_config.uncertainty_threshold`, silently overwriting any default values to align the backend perfectly with the user's high-level request. Similar scaling functions will be applied to the `loop_strategy.max_iterations` and `replay_buffer_size`.

**Step 3: Building the FastAPI Gateway (`src/api/main.py`, `src/api/routes.py`)**
We will initialize the FastAPI application in `main.py`, configuring standard CORS middleware to permit connections from the local React frontend development server. In `routes.py`, we will define a robust `POST /config/submit` endpoint. The endpoint signature will demand a top-level `ProjectConfig` object. FastAPI, integrated seamlessly with Pydantic, will automatically handle the massive JSON-to-Python parsing, type coercion, and validation tree execution. The endpoint handler itself will be remarkably thin; it will simply return a success HTTP 200 message alongside the serialized, fully resolved configuration back to the client, confirming that the translation was mathematically successful and accepted.

**Step 4: Centralizing Validation Utilities for Architectural Cleanliness**
To maintain adherence to the strict Ruff `C901` (McCabe complexity < 10) constraints, we will extract any duplicated path traversal or regex-based security checks currently living as class methods in `config.py` into a shared, module-level private function (e.g., `_validate_string_security(val: str)` inside `src/dynamics/security_utils.py`). This crucial architectural refactoring ensures that the new GUI schemas can reuse the exact same rigorous security logic without massive code duplication or shadowing, resulting in a significantly cleaner and more maintainable validation tier.

## 5. Test Strategy

Testing this cycle focuses exclusively on mathematical correctness, structural schema validation, and API routing behaviors, with absolute, strict isolation from the heavy physical computation engines to ensure sub-second test execution.

**Unit Testing Approach:**
-   **Target:** The `@model_validator` functions embedded within `config.py`.
-   **Method:** We will programmatically instantiate a `ProjectConfig` object passing an `intent` dictionary set to `accuracy_speed_tradeoff = 1` (Maximum Speed). We will then assert that the resulting deeply nested `DistillationConfig.uncertainty_threshold` exactly matches the expected high floating-point value (e.g., 0.15). We will repeat this exact test for the edge cases `accuracy_speed_tradeoff = 10` (Maximum Accuracy) and the median `5`.
-   **Target:** `WorkflowIntentConfig` boundary enforcement.
-   **Method:** We will attempt to instantiate the intent configuration model with an `accuracy_speed_tradeoff` of `0` or `11`. We will assert that a native Pydantic `ValidationError` is immediately raised, thoroughly verifying that the input boundary logic is physically and mathematically sound.
-   **Isolation Guarantee:** These unit tests will execute purely in memory. They require absolute zero file I/O, no temporary directories, and zero mocking of external binaries, as they strictly test the Pydantic instantiation and mathematical translation logic.

**Integration Testing Approach:**
-   **Target:** The FastAPI `POST /config/submit` endpoint and ASGI pipeline.
-   **Method:** We will utilize the highly efficient `fastapi.testclient.TestClient`. We will construct a complete, nested JSON dictionary representing a realistic, heavy GUI payload originating from the React frontend. We will issue the synchronous POST request.
-   **Assert:** We will verify that the HTTP status code returned is precisely 200 OK. We will parse the JSON response body and verify that the deeply nested parameters (which were specifically omitted from the initial request payload but dynamically generated by the intent translation validator) are present, correctly typed, and mathematically accurate in the response.
-   **Target:** Endpoint Error Handling and Security Rejection.
-   **Method:** We will send a JSON payload equipped with a malicious string containing known path traversals (e.g., `"target_material": "../../etc/passwd"` or deeply nested JSON bombs).
-   **Assert:** We will verify that the FastAPI application automatically catches the deeply nested Pydantic validation error and returns an HTTP 422 Unprocessable Entity status code. We will assert that the error payload contains a clear message detailing the security violation, thereby guaranteeing that malicious payloads can never reach the sensitive orchestration layers.
