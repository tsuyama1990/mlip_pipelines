# CYCLE03: Auto-HPO and Base MLIP Setup UI Specification

## 1. Summary

CYCLE03 aims to implement the highly critical "MLIP Setup and Auto-HPO (Hyperparameter Optimization)" requirements specified in the PRD. Fine-tuning a massive Foundation Model like MACE requires selecting mathematically optimal hyperparameters (e.g., learning rate, energy weight, force weight, batch size) to prevent the well-known issue of catastrophic forgetting, where the model learns the new DFT data but completely unlearns its foundational physics knowledge. Non-experts cannot perform this intricate tuning manually; it requires a PhD-level understanding of gradient descent dynamics.

This cycle resolves this bottleneck by introducing a new, fully automated API-driven Auto-HPO workflow. Instead of exposing numeric text boxes and requiring users to guess learning rates, the GUI will present users with high-level, intent-based policies (e.g., "Generalize to preserve base knowledge" vs. "Specialize purely for this specific interface"). Upon selecting a policy, the backend will execute a bounded Bayesian Optimization loop over a small, curated validation set. The API will then return the results as a clean Pareto front, allowing the frontend to present simple, visual choices (Model A vs. Model B) to the user. This entirely abstracts the most mathematically complex aspect of MLIP deployment.

## 2. System Architecture

This cycle adds sophisticated mathematical optimization logic directly into the `src/api` and `src/trainers` layers, fundamentally expanding the system from a simple orchestrator into an intelligent, self-tuning engine.

**File Structure (ASCII Tree):**
```text
mlip-pipelines/
├── src/
│   ├── api/
│   │   ├── routes.py                   # Extended with crucial /hpo/start and /hpo/status endpoints
│   ├── domain_models/
│   │   ├── config.py                   # Extended TrainerConfig with AutoHPOConfig
│   │   └── gui_schemas.py              # New models specifically for HPO payloads and Pareto results
│   ├── trainers/
│   │   ├── **auto_hpo_manager.py**         # New dedicated Bayesian optimization runner and state manager
└── tests/
    └── trainers/
        └── test_auto_hpo_manager.py    # Comprehensive unit tests for the Bayesian optimization loop
```

The data flow is highly asynchronous. The user initiates the HPO process via an HTTP `POST /hpo/start` endpoint, passing a lightweight `HPOPolicyConfig` payload. The FastAPI backend instantly responds with a 202 Accepted and dispatches this request to the `auto_hpo_manager.py` as a background execution task, ensuring the API server remains responsive. The frontend periodically polls a `GET /hpo/status` endpoint (or relies on WebSockets, implemented in Cycle 05) to receive the dynamically generating Pareto front points. Once the background process concludes and a point is selected by the user on the visual scatter plot, the user submits an HTTP `POST /hpo/select` to irrevocably lock the chosen hyperparameter vector into the active `TrainerConfig` memory space.

## 3. Design Architecture

**Domain Concepts & Pydantic Models:**
-   **`HPOPolicyConfig`**: A strongly typed model residing in `gui_schemas.py` representing the user's high-level tuning intent.
    -   `policy_type`: An Enum strictly limited to (`GENERALIZE`, `SPECIALIZE`, `BALANCED`). This dictates the boundary limits of the search space.
    -   `max_trials`: Integer representing the maximum number of HPO iterations (default: 10, to prioritize speed over exhaustive search).
-   **`AutoHPOConfig`**: Deeply integrated into the core `TrainerConfig` in `config.py` to allow the training engine to know its bounded, mathematically safe search space.
    -   `learning_rate_bounds`: A strict Tuple of floats (e.g., `(1e-4, 1e-2)`).
    -   `energy_weight_bounds`: A strict Tuple of floats governing force-matching priorities.
-   **`HPOResultDTO`**: A robust Data Transfer Object returned to the API representing a single evaluated point on the Pareto front (e.g., `{"model_id": "Candidate_A", "rmse_e": 0.05, "rmse_f": 0.1, "forgetting_score": 0.01}`). This contains all the data required for the frontend to render the interactive scatter plot.

**Key Invariants, Constraints, and Validation Rules:**
1.  **Bounded Execution Time Security**: HPO tasks involving Foundation Models are extraordinarily expensive in terms of GPU hours. The API validation layer must strictly enforce a maximum upper limit on the number of trials (`max_trials <= 20`) to prevent API timeouts, excessive HPC utilization, or malicious Denial of Service (DoS) attacks designed to bankrupt cloud compute budgets.
2.  **Stateless API Design and Caching**: The background HPO process must *never* permanently mutate the master `ProjectConfig` until the user explicitly selects a candidate model via the designated `/hpo/select` endpoint. Intermediate optimization results must be cached ephemerally and safely in memory or a temporary Redis/SQLite store.
3.  **Mathematical Validity**: The bounds configured for learning rates and weights must be rigorously validated to ensure they are strictly positive and that the lower bound never exceeds the upper bound, preventing the Bayesian optimizer from crashing due to malformed mathematical spaces.

## 4. Implementation Approach

The implementation involves constructing a safe, asynchronous wrapper around complex scientific optimization libraries.

**Step 1: Pydantic Schema Definitions and Constraints**
Define the `HPOPolicyConfig` and `HPOResultDTO` schemas within the GUI definitions file. Extend the core `TrainerConfig` to natively accept an `AutoHPOConfig` object containing the numeric search bounds. Implement critical `@model_validator` methods that enforce the strict `max_trials <= 20` limits and ensure that all provided tuple bounds are mathematically sound (lower < upper).

**Step 2: AutoHPO Manager Implementation and Bayesian Logic**
Create the completely new `src/trainers/auto_hpo_manager.py` module. Implement a core function `run_hpo(policy: HPOPolicyConfig)` that utilizes a lightweight, robust Bayesian optimization library (such as `scipy.optimize` or `optuna` if available within the `uv` dependency tree) to intelligently sample the defined, bounded hyperparameter space. For each sequential trial, the function must evaluate the validation set RMSE and, crucially, the "forgetting score" (the measured error on the original foundation model's training data subset). It must track the best performing models and construct the mathematical Pareto front.

**Step 3: FastAPI Endpoint Integration and Async Tasks**
In `src/api/routes.py`, construct the `POST /hpo/start` endpoint. This endpoint takes the parsed `HPOPolicyConfig` payload. Because HPO is inherently slow and blocking, it *must* be executed utilizing `fastapi.BackgroundTasks` or a dedicated thread pool. The endpoint will immediately return a generated `hpo_task_id` string.
Implement the corresponding `GET /hpo/status/{task_id}` endpoint to query the ephemeral cache and return the currently discovered Pareto front of `HPOResultDTO` objects.
Finally, implement the `POST /hpo/select` endpoint to take the user-chosen `HPOResultDTO` and decisively update the active `TrainerConfig` stored in the server's memory, preparing it for the main orchestrator run.

## 5. Test Strategy

Testing this cycle focuses heavily on robust API orchestration, strict optimization bounds enforcement, and asynchronous state management, deliberately isolating and mocking the slow, heavy ML training components to ensure continuous integration pipelines remain blazingly fast.

**Unit Testing Approach:**
-   **Target:** The `run_hpo` Bayesian optimization loop and bounds enforcement.
-   **Method:** We will heavily mock the actual MACE training function, replacing it with a dummy function designed to return an instantly calculated dummy RMSE and forgetting score based on the input hyperparameters via a deterministic, multi-dimensional polynomial function.
-   **Assert:** Verify that the HPO loop correctly and intelligently samples the multi-dimensional space entirely within the defined `learning_rate_bounds` and successfully identifies the global minimum of the dummy polynomial function. Assert categorically that exactly `max_trials` evaluations occur, and not a single one more.

**Integration Testing Approach:**
-   **Target:** The `/hpo/start` and `/hpo/status` background task lifecycle and REST routing.
-   **Method:** Submit a valid `POST /hpo/start` payload explicitly setting `max_trials=2`. Utilize `FastAPI.testclient.TestClient`.
-   **Assert:** Ensure an immediate, non-blocking 202 Accepted response is received. Periodically poll the `/hpo/status` endpoint. Ensure that the final status payload contains a cleanly populated array of `HPOResultDTO` objects representing the completed Pareto front, proving the background task completed successfully without the primary web server endpoint timing out.
-   **Target:** DoS Prevention. Submit a payload requesting `max_trials=5000` and assert an immediate 422 Unprocessable Entity is returned.

**Side-effect Isolation:** The incredibly heavy Machine Learning libraries (`mace-torch`, `torch`) will be entirely mocked at the deepest `sys.modules` level to ensure tests run in milliseconds rather than hours and do not require GPU hardware. The ephemeral state of the HPO manager will utilize a simple, garbage-collected in-memory dictionary keyed by `task_id` to prevent disk I/O bottlenecks.
