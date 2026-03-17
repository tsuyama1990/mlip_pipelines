# CYCLE05: Real-time Monitoring & Pre-flight Diagnostics Specification

## 1. Summary

CYCLE05 delivers two highly critical User Experience (UX) features outlined exhaustively in the PRD: The Real-time Monitoring Dashboard and the mandatory "Run 0" Pre-flight Validation check. These two features transform the system from a batch-processing black box into a highly interactive, immediately responsive platform.

The Run 0 check is an incredibly crucial fail-fast mechanism. Before the user ever commits to a potentially expensive, multi-day compute job on an HPC cluster—which could burn thousands of node-hours unnecessarily—the backend performs an instantaneous, zero-step molecular dynamics initialization. This mechanism instantly and decisively catches common, fatal structural errors like unphysical atomic overlaps (which immediately cause infinite forces and catastrophic energy divergence) or subtle syntax errors accidentally generated in the complex LAMMPS scripts. By failing in seconds instead of hours, the UX is vastly improved.

The Real-time Monitoring Dashboard transforms the traditionally static, unreadable command-line output of the Orchestrator into a live, highly visual streaming data feed. Using incredibly fast WebSockets, the backend will actively and asynchronously push training loss curves from the ML models, real-time potential energy maps during the MD exploration, and critical Active Learning events (like precisely when a DFT calculation is triggered due to high epistemic uncertainty) directly to the frontend GUI, allowing users to visually track the mathematical health of their highly complex machine learning models in real-time.

## 2. System Architecture

This cycle focuses heavily on the `src/api` layer, specifically the complex WebSocket implementation, and introduces deep, asynchronous telemetry hooks into the core computation engines to cleanly stream their internal mathematical metrics without ever blocking the primary physics calculations.

**File Structure (ASCII Tree):**
```text
mlip-pipelines/
├── src/
│   ├── api/
│   │   ├── routes.py                   # Extended with the crucial /validate/run0 pre-flight endpoint
│   │   └── **websocket.py**                # New highly concurrent WebSocket Connection Manager
│   ├── core/
│   │   ├── orchestrator.py
│   │   └── **telemetry_hook.py**           # New asynchronous metrics broadcasting hook and singleton
│   ├── dynamics/
│   │   └── dynamics_engine.py          # Modified to emit Run0 diagnostic data and live MD thermodynamic stats
│   ├── trainers/
│   │   └── auto_hpo_manager.py         # Modified to stream optimization loss curves
└── tests/
    └── api/
        └── test_websocket.py           # Rigorous integration tests for non-blocking telemetry streaming and Run 0 parsing
```

The system flow requires that the `Run 0` validation occurs via a standard, synchronous HTTP POST to `/validate/run0`. The backend temporarily, safely instantiates the heavy `DynamicsEngine` with a specific `run 0` override command, aggressively parses the immediate stdout output (checking for specific LAMMPS crash codes, "Lost atoms" messages, or infinite `NaN` energies), and returns a highly detailed boolean pass/fail payload with specific diagnostic strings.
For the Real-time Monitoring, the frontend establishes a persistent, stateful WebSocket connection to `ws://.../ws/telemetry`. Throughout the execution of the massive `orchestrator.run_cycle()`, sub-components like the highly complex `PacemakerTrainer` or the `DynamicsEngine` will constantly invoke a global singleton `TelemetryBroadcaster.emit()`. This sophisticated broadcaster pushes serialized JSON packets (containing accurate timestamps, specific metric names, and floating-point values) down the open WebSocket to all connected GUI clients, instantly updating their React dashboards.

## 3. Design Architecture

**Domain Concepts & Pydantic Models:**
-   **`RunZeroDiagnosticsDTO`**: A robust model defined in `gui_schemas.py` that fully describes the immediate health of the structure.
    -   `passed`: A definitive Boolean.
    -   `errors`: A List of strings detailing exactly what failed (e.g., ["Fatal atomic collision detected between IDs 5 and 12", "LAMMPS Syntax Error on line 42 regarding invalid fix arguments"]).
    -   `initial_energy`: An optional Float providing the immediate baseline energy of the system.
-   **`TelemetryPacket`**: A highly structured model representing a streamable, real-time data point.
    -   `timestamp`: Float (precise epoch time).
    -   `source`: An Enum definitively indicating the originating subsystem (`TRAINER`, `DYNAMICS`, `ORACLE`).
    -   `metric_type`: An Enum defining the exact data type (`LOSS`, `ENERGY`, `UNCERTAINTY_MAX`, `EVENT`).
    -   `value`: A highly flexible Float or String payload containing the actual data point.

**Key Invariants, Constraints, and Validation Rules:**
1.  **Absolutely Non-Blocking Telemetry**: The crucial act of emitting a `TelemetryPacket` must absolutely never, under any circumstances, block the main computational physics threads. If a WebSocket client abruptly disconnects due to a closed laptop or terrible wifi, the multi-million atom Orchestrator must not crash, hang, or even slow down. The `TelemetryBroadcaster` must strictly use asynchronous `asyncio.Queue` structures and broad `contextlib.suppress` wrappers to handle broken pipes and disconnected sockets with extreme grace.
2.  **Strict Run 0 Environmental Isolation**: The `Run 0` diagnostic test must execute in a completely isolated, ephemeral temporary directory provided by Python's `tempfile.TemporaryDirectory`. It must aggressively and successfully clean up any generated binaries, massive `.xyz` files, or `log.lammps` files immediately after completion—regardless of whether the LAMMPS run crashed or succeeded—to absolutely prevent polluting the actual project workspace and exhausting server disk space.

## 4. Implementation Approach

**Step 1: Concurrent WebSocket Connection Manager**
Create the entirely new `src/api/websocket.py` module. Implement a robust `ConnectionManager` class strictly following FastAPI's asynchronous documentation. It will maintain a thread-safe list of active `WebSocket` connections and securely provide `async` methods like `broadcast(packet: str)` to disseminate the JSON data to all connected React clients.

**Step 2: Non-Blocking Telemetry Hook Implementation**
Create the `src/core/telemetry_hook.py` module. Implement a thread-safe singleton `TelemetryBroadcaster` that utilizes Python's highly efficient `asyncio.Queue`. The deeply nested core engines (e.g., within `src/trainers/`) will call a very fast, synchronous `emit()` method which instantly places data onto the queue. A persistent background task residing in the FastAPI application will continuously and asynchronously read from this queue and rapidly call `ConnectionManager.broadcast()`, completely decoupling the heavy physics math from the network I/O.

**Step 3: Integrating Telemetry into Massive Core Engines**
Carefully modify `dynamics_engine.py` to continuously parse the `log.lammps` output file periodically (or utilize efficient LAMMPS C++ python library hooks if available) and `emit()` the current timestep's exact potential energy and the foundation model's maximum atomic epistemic uncertainty. Modify the incredibly complex ML trainer classes to cleanly `emit()` the exact training and validation loss calculations precisely after every single epoch completes.

**Step 4: Run 0 Pre-flight Diagnostic Endpoint Logic**
In `routes.py`, construct the robust `POST /validate/run0` endpoint accepting a complete `ProjectConfig`. In the handler, securely extract the initial structure, generate a custom LAMMPS script specifically replacing the standard `run N` command with `run 0`, and securely execute the LAMMPS binary in a subprocess. Aggressively parse the standard output and standard error streams. If LAMMPS returns any non-zero exit code, or if the parsed thermodynamic output shows the string "NaN" or wildly positive energy values (firmly indicating extreme atomic overlap), immediately populate and return a failed `RunZeroDiagnosticsDTO` without initiating the orchestrator.

## 5. Test Strategy

Testing this deeply interactive cycle focuses intensely on verifying asynchronous queue management, testing resilience against dropped network connections, and accurately parsing highly complex physics-engine crash states from raw text streams.

**Unit Testing Approach:**
-   **Target:** The `Run 0` stdout parser.
-   **Method:** We will feed predefined, highly specific string blocks mimicking actual LAMMPS crash logs (e.g., "ERROR: Lost atoms: original 100 current 99" or "Step Temp E_pair ... 0 300 NaN"). We will assert that the parser flawlessly and correctly flags these as failures and accurately extracts the relevant error message without crashing.
-   **Target:** The `TelemetryBroadcaster` strict non-blocking behavior.
-   **Method:** We will instantiate the broadcaster singleton. We will rapidly call `emit()` exactly 10,000 times in a tight loop. We will assert that the `asyncio` queue fills appropriately without ever blocking or slowing down the main thread, proving it is safe for high-performance physics loops.

**Integration Testing Approach:**
-   **Target:** The critical `/ws/telemetry` WebSocket endpoint.
-   **Method:** We will use `FastAPI.testclient.TestClient.websocket_connect`. We will connect a dummy client. In a completely separate, isolated thread, we will trigger an `emit()` call from the mock ML trainer. We will categorically assert that the test client successfully receives the exact JSON string representation of the `TelemetryPacket` perfectly intact over the active socket. Crucially, we will disconnect the client abruptly, emit another packet, and assert the backend system handles the broken pipe Exception beautifully and gracefully without crashing the server.
-   **Side-effect Isolation:** The Run 0 LAMMPS execution tests will heavily mock the core `subprocess.run` call, explicitly returning predefined `stdout` byte strings that simulate crashes, absolutely preventing actual, slow binary execution and ensuring the test suite remains rapid, deterministic, and highly portable.
