# CYCLE04: Active Learning (OTF) Smart Control Integration UAT Plan

## 1. Test Scenarios

### Scenario ID: UAT-04-01: Orchestrator Asynchronous Start & Status Tracking
**Priority:** High
**Description:** Verify that submitting a "START" command to the API successfully and securely launches the core Orchestrator process entirely asynchronously in the background. The primary REST API must absolutely not block, hang, or timeout while the massive HPC simulation runs. Furthermore, subsequent calls from the React frontend to the `/orchestrator/status` polling endpoint must accurately, reliably, and rapidly reflect the Orchestrator's current iteration and specific active phase (e.g., transitioning seamlessly from "PHASE2_VALIDATION" into "PHASE3_MD_EXPLORATION") as it mathematically progresses through the complex Active Learning loop.

### Scenario ID: UAT-04-02: Graceful Pause and Stateful Resume
**Priority:** Critical
**Description:** Verify that sending an asynchronous "PAUSE" command to a heavily running Orchestrator definitively triggers a graceful, deterministic shutdown at the absolute nearest mathematically safe boundary (e.g., securely between the massive MD exploration phase and the delicate Foundation Model Finetuning phase), rather than instantly killing the process and completely corrupting the SQLite state database and binary restart files. After definitively confirming the status is explicitly "PAUSED", sending a subsequent "RESUME" command must successfully and seamlessly restart the Orchestrator from the exact phase, iteration, and precise atomic geometry where it previously left off.

### Scenario ID: UAT-04-03: Concurrency Control and Lock Management
**Priority:** High
**Description:** Verify that the highly secure `task_manager` strictly and perfectly prevents multiple Orchestrator instances from running concurrently in the exact same project directory. Because the system writes to local SQLite databases and generates massive temporary files, a race condition here is fatal. If an Orchestrator is already officially "RUNNING", any new, malicious, or accidental "START" command for that precise project must be instantaneously rejected with a highly descriptive HTTP error (e.g., 409 Conflict) to categorically prevent devastating SQLite database corruption and data loss.

## 2. Behavior Definitions

### UAT-04-01: Orchestrator Asynchronous Start & Status Tracking
**GIVEN** a running, fully isolated instance of the Adaptive-MLIP FastAPI backend loaded with a perfectly valid configuration payload
**WHEN** the extensive payload and a decisive `START` command are submitted via HTTP POST to the `/orchestrator/command` endpoint
**THEN** the API system immediately and flawlessly responds with an HTTP 202 Accepted status
**AND** the massive core Orchestrator safely begins executing cycles completely in the background without blocking the main event loop
**WHEN** a standard HTTP GET request is made to the `/orchestrator/status` endpoint exactly five seconds later
**THEN** the serialized JSON status payload correctly and accurately returns `{"status": "RUNNING", "current_iteration": 1, "current_phase": "PHASE3_MD_EXPLORATION"}`.

### UAT-04-02: Graceful Pause and Stateful Resume
**GIVEN** an active, extremely busy Orchestrator running heavily in the background, currently executing at exactly Iteration 2, Phase 1
**WHEN** a decisive `PAUSE` command is submitted via the REST API to `/orchestrator/command`
**THEN** the API instantly returns an HTTP 202 Accepted status indicating the signal was received
**AND** the Orchestrator finishes its current complex mathematical phase and halts flawlessly at the boundary
**WHEN** aggressively polling `/orchestrator/status` finally returns `{"status": "PAUSED", "current_iteration": 2, "current_phase": "PHASE2_VALIDATION"}`
**WHEN** a subsequent `RESUME` command is definitively submitted via the API
**THEN** the Orchestrator safely restarts from the database and immediately begins executing exactly `PHASE2_VALIDATION` precisely at Iteration 2, dropping absolutely zero physical data.

### UAT-04-03: Concurrency Control and Lock Management
**GIVEN** an active, highly demanding Orchestrator running safely in the background exclusively for the directory `Project_A`
**WHEN** a brand new, potentially conflicting `START` command is submitted to the `/orchestrator/command` endpoint targeting the exact same `Project_A` directory path
**THEN** the robust backend system immediately and categorically rejects the incoming payload
**AND** responds securely with a definitive HTTP 409 Conflict status
**AND** the JSON error details explicitly and clearly indicate that an active run lock definitively exists for this directory and cannot be bypassed
**AND** the original, highly complex Orchestrator run continues its mathematical operations completely uninterrupted and completely unaware of the failed secondary attempt.
