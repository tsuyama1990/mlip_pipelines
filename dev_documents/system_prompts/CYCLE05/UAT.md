# CYCLE05: Real-time Monitoring & Pre-flight Diagnostics UAT Plan

## 1. Test Scenarios

### Scenario ID: UAT-05-01: Run 0 Instant Validation of Severely Malformed Structures
**Priority:** High
**Description:** Verify conclusively that the backend system instantly, reliably, and gracefully catches severely unphysical initial atom configurations (e.g., severe atomic spatial overlaps causing mathematically infinite repulsive forces) before the user ever attempts to initiate a costly, multi-day massive orchestrator run heavily utilizing an HPC cluster. The React frontend should rapidly submit a complex data structure completely with intentionally colliding atoms directly to the highly secure `/validate/run0` endpoint, and the backend must flawlessly execute an incredibly fast zero-step physics evaluation, cleanly parse the resulting catastrophic mathematical energy divergence directly from the LAMMPS binary's output, and securely return a perfectly clear, actionable failure diagnostic identifying the exact nature and coordinates of the collision, brilliantly saving hours of wasted compute time and cluster queue positioning.

### Scenario ID: UAT-05-02: Run 0 Complex Syntax Error Detection
**Priority:** High
**Description:** Verify conclusively that the intelligent system correctly, instantly, and perfectly identifies complex syntactic errors secretly generated inside the immensely complex LAMMPS scripts prior to any actual, dangerous simulation execution on the cluster. The frontend cleanly submits a massive configuration payload secretly containing a deliberately invalid geometric region name or malformed, incorrect `fix` integration arguments. The highly robust backend's Run 0 check should decisively and instantly fail immediately upon LAMMPS binary initialization, flawlessly catching the obtuse syntax error code and securely, beautifully returning the specific, exact failing script line directly back to the GUI for the user to quickly correct without ever needing to read the raw `log.lammps` file.

### Scenario ID: UAT-05-03: Real-Time Telemetry Streaming and Absolute Fault Tolerance
**Priority:** Critical
**Description:** Verify conclusively that the highly concurrent WebSocket connection (beautifully hosted at `/ws/telemetry`) successfully, continuously, and securely streams massive live metrics (e.g., ML training loss curves, dynamic MD potential energy values, and absolutely critical Active Learning DFT trigger events) from the intensely active backend python orchestrator directly to a connected frontend web client. The massive, continuous stream of data must absolutely arrive in perfectly correct chronological order and be properly, cleanly formatted as strictly serialized JSON packets. Furthermore, the master orchestrator must absolutely not crash, hang, pause, or stutter if the active WebSocket client gracefully or abruptly disconnects entirely (flawlessly simulating a closed browser tab or a severely dropped VPN connection), conclusively proving the massive system is perfectly fault-tolerant, resilient, and robust for enterprise deployment.

### Scenario ID: UAT-05-04: Hybrid Force Field Intelligent Compatibility Diagnosis
**Priority:** High
**Description:** Verify flawlessly that the pre-flight Run 0 diagnostic validation endpoint perfectly intercepts and intelligently analyzes the atomic composition of the user's uploaded structure directly against the explicitly configured classical force field parameters (e.g., EAM, MEAM, or simple Lennard-Jones parameters) natively residing within the `ProjectConfig`. If the highly intelligent validator categorically detects that specific, vital atomic species (e.g., Nickel in a Platinum-Nickel binary alloy) are completely missing absolutely required parameter definitions, the system must instantly abort the Run 0 execution gracefully, perfectly avoiding a fatal, obscure LAMMPS C++ segmentation fault, and securely return a highly descriptive HTTP 422 error explicitly listing the exact missing elemental symbols cleanly to the user interface.

## 2. Behavior Definitions

### UAT-05-01: Run 0 Instant Validation of Severely Malformed Structures
**GIVEN** a flawlessly running, healthy instance of the Adaptive-MLIP FastAPI backend web server
**AND** a massive, simulated GUI JSON payload meticulously defining a highly complex structure but secretly containing two heavy metal atoms placed exactly and completely at identical spatial coordinates (0, 0, 0)
**WHEN** the massive payload is submitted securely via HTTP POST to the highly guarded `/validate/run0` endpoint
**THEN** the backend system immediately and safely runs a zero-step LAMMPS initialization entirely within a fully isolated, strictly ephemeral temporary directory
**AND** responds almost instantly and flawlessly with an HTTP 200 OK status securely containing a cleanly populated `RunZeroDiagnosticsDTO`
**AND** the comprehensive DTO categorically and explicitly indicates `passed: false`
**AND** the `errors` massive string list cleanly and perfectly contains a highly descriptive string stating "infinite potential energy" or "severe atomic collision detected", permanently preventing further disastrous execution.

### UAT-05-02: Run 0 Complex Syntax Error Detection
**GIVEN** a perfectly running, highly secure instance of the Adaptive-MLIP FastAPI backend
**AND** a massive, simulated GUI JSON payload deliberately containing a syntactically invalid argument explicitly for a LAMMPS group selection block (e.g., improperly passing a raw string instead of the rigidly required integer IDs)
**WHEN** the entirely malformed payload is aggressively submitted to the `/validate/run0` endpoint for processing
**THEN** the robust backend heavily attempts to parse the massive configuration and safely execute the generated Run 0 diagnostic script
**AND** responds definitively and instantly with a completely populated `RunZeroDiagnosticsDTO` explicitly indicating `passed: false`
**AND** the comprehensive `errors` list seamlessly and perfectly contains the exact, specific "ERROR" line text cleanly and beautifully extracted entirely from the LAMMPS C++ binary's standard output error stream, making complex debugging utterly trivial for the non-expert user.

### UAT-05-03: Real-Time Telemetry Streaming and Absolute Fault Tolerance
**GIVEN** an extremely active, highly utilized Orchestrator running heavily and continuously in the background (having been successfully started via the `/orchestrator/command` endpoint)
**WHEN** a web client successfully and perfectly establishes a highly persistent, stateful WebSocket connection directly to the `/ws/telemetry` endpoint
**THEN** the web client immediately begins receiving a rapid, massive stream of perfectly JSON-formatted `TelemetryPacket` strings
**AND** the continuous packets correctly, sequentially, and beautifully show updating `LOSS` metrics emitted smoothly and cleanly by the incredibly complex mock `PacemakerTrainer` machine learning engine
**WHEN** the web client violently, abruptly, and unexpectedly closes the WebSocket connection completely (creating a severe and massive broken pipe TCP exception)
**THEN** the deeply nested, highly active Orchestrator resolutely and flawlessly continues running completely, perfectly unaffected by the massive network error
**AND** absolutely no unhandled exceptions, massive memory leaks, or segmentation crashes occur anywhere within the incredibly complex backend's main Python physics thread, proving absolute enterprise resilience.

### UAT-05-04: Hybrid Force Field Intelligent Compatibility Diagnosis
**GIVEN** a perfectly running instance of the massive Adaptive-MLIP FastAPI backend architecture
**AND** a highly complex, simulated GUI JSON payload specifically defining an ASE structure that explicitly contains both `Pt` and `Ni` atoms natively, perfectly alongside a complex `ProjectConfig` perfectly containing only classical `EAM` parameters specifically targeting the `Pt` atoms
**WHEN** the deeply flawed payload is submitted via HTTP POST cleanly to the `/validate/run0` endpoint for immediate pre-flight diagnostics
**THEN** the backend's highly intelligent `HybridForceFieldValidator` safely intercepts the validation flow and instantly detects the glaring logical omission
**AND** responds beautifully and gracefully with an HTTP 422 Unprocessable Entity status code directly to the client
**AND** the fully populated JSON error details definitively and explicitly list `["Ni"]` strictly within the `missing_forcefield_params` array
**AND** the massive C++ LAMMPS executable is absolutely, completely never invoked, completely avoiding the catastrophic, unreadable segmentation fault that would inevitably follow from attempting to run physics operations on utterly undefined particle interactions.
