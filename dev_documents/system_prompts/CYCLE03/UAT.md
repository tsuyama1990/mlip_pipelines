# CYCLE03: Auto-HPO and Base MLIP Setup UI UAT Plan

## 1. Test Scenarios

### Scenario ID: UAT-03-01: Intent-Driven Auto-HPO Initiation
**Priority:** High
**Description:** Verify that the system correctly initiates an asynchronous background Hyperparameter Optimization (HPO) task when a high-level intent policy ("GENERALIZE" or "SPECIALIZE") is submitted via the REST API. The GUI must not pass raw `learning_rate` or `energy_weight` numerical bounds, as the user is not expected to know them. The backend must independently interpret the policy, mathematically set the boundaries of the Bayesian search space, and begin the complex optimization loop asynchronously without ever blocking the main web server event loop.

### Scenario ID: UAT-03-02: HPO Pareto Front Retrieval and Selection
**Priority:** High
**Description:** Verify that the frontend application can successfully and repeatedly poll the HPO task status endpoint and retrieve the dynamically generated Pareto front of model candidates (e.g., Model A vs Model B), containing their respective evaluated RMSE and calculated "forgetting scores". Finally, verify that submitting a final selection via the `/hpo/select` endpoint correctly, securely, and permanently updates the active `TrainerConfig` within the backend's memory state, readying it for the main simulation.

### Scenario ID: UAT-03-03: Security Against Excessive HPO Trials (DoS Prevention)
**Priority:** Critical
**Description:** Verify that the backend strictly and aggressively rejects any API payload attempting to set the `max_trials` parameter of the HPO process to an excessively high number (e.g., > 20). Because GPU computing time is extraordinarily expensive, this security measure is paramount to prevent malicious, misconfigured, or runaway GUI requests from monopolizing the host HPC resources, exhausting the computing budget, and causing a systemic Denial of Service.

## 2. Behavior Definitions

### UAT-03-01: Intent-Driven Auto-HPO Initiation
**GIVEN** a healthy, running instance of the Adaptive-MLIP FastAPI backend with a mock validation training dataset loaded into memory
**AND** a simulated GUI JSON payload containing a cleanly formatted `HPOPolicyConfig` specifying `policy_type: GENERALIZE` and `max_trials: 5`
**WHEN** the JSON payload is submitted via HTTP POST to the `/hpo/start` endpoint
**THEN** the system immediately and without hesitation responds with an HTTP 202 Accepted status
**AND** the JSON response returns a newly generated, unique `task_id` string
**AND** a background asynchronous task successfully begins executing exactly 5 iterations of the highly complex Bayesian optimization loop without locking the main API thread.

### UAT-03-02: HPO Pareto Front Retrieval and Selection
**GIVEN** an active HPO background task identified by `task_id_123` that has successfully completed its grueling optimization loop
**WHEN** a standard HTTP GET request is made to the `/hpo/status/task_id_123` endpoint
**THEN** the system immediately responds with an HTTP 200 OK status
**AND** the JSON payload contains a fully populated list of `HPOResultDTO` objects representing the calculated Pareto front
**WHEN** a subsequent HTTP POST request is made to the `/hpo/select` endpoint providing the payload `model_id: "Model_A"`
**THEN** the backend's active `TrainerConfig`'s deeply nested `learning_rate` and `energy_weight` parameters are permanently updated to match the mathematically optimal parameters discovered for "Model_A".

### UAT-03-03: Security Against Excessive HPO Trials (DoS Prevention)
**GIVEN** a running instance of the Adaptive-MLIP FastAPI backend exposed to the network
**AND** a simulated GUI JSON payload containing an `HPOPolicyConfig` maliciously or erroneously attempting to request `max_trials: 500`
**WHEN** the payload is submitted via HTTP POST to the `/hpo/start` endpoint
**THEN** the system's Pydantic validation layer immediately intercepts and rejects the payload before it reaches the orchestrator
**AND** responds with a definitive HTTP 422 Unprocessable Entity status code
**AND** the error details explicitly indicate that `max_trials` must be strictly less than or equal to 20 to protect system resources
**AND** absolutely no background task is initiated, preserving the HPC cluster's compute budget.
