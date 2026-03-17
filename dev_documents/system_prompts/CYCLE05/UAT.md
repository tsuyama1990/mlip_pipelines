# CYCLE05: Real-time Monitoring & Pre-flight Diagnostics UAT Plan

## 1. Test Scenarios

### Scenario ID: UAT-05-01: Run 0 Instant Validation of Malformed Structures
**Priority:** High
**Description:** Verify that the system instantly and reliably catches severely unphysical initial configurations (e.g., severe atomic overlaps causing mathematically infinite forces) before the user initiates a costly, multi-day orchestrator run on an HPC cluster. The frontend should submit a structure with intentionally colliding atoms directly to the `/validate/run0` endpoint, and the backend must execute an incredibly fast zero-step evaluation, cleanly parse the resulting catastrophic energy divergence from the LAMMPS binary, and return a perfectly clear failure diagnostic identifying the exact nature of the collision, saving hours of wasted compute time.

### Scenario ID: UAT-05-02: Run 0 Syntax Error Detection
**Priority:** High
**Description:** Verify that the system correctly, instantly identifies syntactic errors generated in the immensely complex LAMMPS scripts prior to any actual simulation execution. The frontend submits a payload containing a deliberately invalid region name or malformed `fix` arguments. The backend's Run 0 check should decisively fail immediately upon LAMMPS binary initialization, flawlessly catching the syntax error code and securely returning the specific failing script line back to the GUI for the user to correct.

### Scenario ID: UAT-05-03: Real-Time Telemetry Streaming and Fault Tolerance
**Priority:** Critical
**Description:** Verify that the highly concurrent WebSocket connection (hosted at `/ws/telemetry`) successfully, continuously streams live metrics (e.g., ML training loss, dynamic MD potential energy, and critical DFT trigger events) from the intensely active backend orchestrator directly to a connected frontend client. The massive stream of data must arrive in correct chronological order and be properly formatted as serialized JSON packets. Furthermore, the orchestrator must absolutely not crash, hang, or pause if the WebSocket client disconnects abruptly (simulating a closed browser tab or a dropped VPN connection), proving the system is fault-tolerant and robust.

## 2. Behavior Definitions

### UAT-05-01: Run 0 Instant Validation of Malformed Structures
**GIVEN** a running, healthy instance of the Adaptive-MLIP FastAPI backend
**AND** a simulated GUI JSON payload meticulously defining a structure with two heavy atoms placed exactly at coordinates (0, 0, 0)
**WHEN** the payload is submitted via HTTP POST to the highly secure `/validate/run0` endpoint
**THEN** the backend system immediately runs a zero-step LAMMPS initialization in a fully isolated, ephemeral temporary directory
**AND** responds almost instantly with an HTTP 200 OK status containing a populated `RunZeroDiagnosticsDTO`
**AND** the DTO categorically indicates `passed: false`
**AND** the `errors` string list cleanly contains a descriptive string stating "infinite potential energy" or "severe atomic collision detected", preventing further execution.

### UAT-05-02: Run 0 Syntax Error Detection
**GIVEN** a running instance of the Adaptive-MLIP FastAPI backend
**AND** a simulated GUI JSON payload containing a syntactically invalid argument for a LAMMPS group selection (e.g., passing a string instead of the required integer IDs)
**WHEN** the malformed payload is submitted to the `/validate/run0` endpoint
**THEN** the backend attempts to parse the configuration and execute the generated Run 0 script
**AND** responds definitively with a `RunZeroDiagnosticsDTO` indicating `passed: false`
**AND** the `errors` list seamlessly contains the exact "ERROR" line text cleanly extracted from the LAMMPS binary's standard output stream, making debugging trivial for the user.

### UAT-05-03: Real-Time Telemetry Streaming and Fault Tolerance
**GIVEN** an active, highly utilized Orchestrator running heavily in the background (successfully started via `/orchestrator/command`)
**WHEN** a client successfully establishes a persistent WebSocket connection to the `/ws/telemetry` endpoint
**THEN** the client immediately begins receiving a rapid stream of JSON-formatted `TelemetryPacket` strings
**AND** the packets correctly, sequentially show updating `LOSS` metrics emitted smoothly by the mock `PacemakerTrainer`
**WHEN** the client abruptly and unexpectedly closes the WebSocket connection (creating a severe broken pipe exception)
**THEN** the active Orchestrator resolutely continues running completely unaffected by the network error
**AND** absolutely no unhandled exceptions, memory leaks, or crashes occur anywhere in the backend's main Python thread.
