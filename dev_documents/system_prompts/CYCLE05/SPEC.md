# CYCLE05: Real-time Monitoring & Pre-flight Diagnostics Specification

## 1. Summary

CYCLE05 introduces the incredibly vital "Pre-flight Check" (referred to as Run 0 Validation), the deployment of a highly concurrent WebSocket streaming telemetry architecture, and the newly defined, critical Hybrid Force Field diagnostic tool. These massive features transform the system from a static batch-processing black box into a highly interactive, beautifully responsive, deeply intelligent platform.

The Run 0 check is an incredibly crucial fail-fast mechanism. Before the user ever commits to a potentially expensive, multi-day compute job on an HPC cluster—which could silently burn thousands of node-hours unnecessarily—the backend rigorously performs an instantaneous, zero-step molecular dynamics initialization. This sophisticated mechanism instantly and decisively catches common, fatal structural topology errors like unphysical atomic overlaps (which immediately cause infinite forces and catastrophic energy divergence) or incredibly subtle syntax errors accidentally generated in the massive, complex LAMMPS scripts. By failing in seconds instead of hours, the UX is vastly, immeasurably improved.

Furthermore, the introduction of the Hybrid Force Field compatibility script automatically and securely scans the incoming loaded atomic structure. It cross-references the specific atomic species present natively within the `.extxyz` file against any classical empirical pair-potentials provided manually by the user in the configuration payload. If a user provides an EAM potential explicitly for Platinum but completely forgets the Nickel parameters in a highly complex Pt-Ni alloy simulation, the validator will immediately and flawlessly catch this specific omission and fail the pre-flight check, intelligently suggesting parameter completion. This completely removes the obscure and unhelpful LAMMPS segmentation faults that typically plague new users attempting hybrid classical/ML simulations.

The Real-time Monitoring Dashboard completely transforms the traditionally static, unreadable, and frustrating command-line output of the Orchestrator into a live, highly visual streaming JSON data feed. Using incredibly fast, non-blocking WebSockets, the robust backend will actively and asynchronously push massive training loss curves strictly from the ML models, real-time potential energy maps generated during the MD exploration, and critical Active Learning events (like precisely when a heavy DFT calculation is triggered due to high epistemic uncertainty limits) directly to the frontend GUI, beautifully allowing users to visually track the mathematical health of their highly complex machine learning models perfectly in real-time.

## 2. System Architecture

This massive cycle focuses incredibly heavily on the `src/api` layer, specifically implementing the complex WebSocket Connection Manager, and introduces deep, asynchronous telemetry hooks natively into the core Python computation engines to cleanly stream their internal mathematical metrics without ever, under any circumstances, blocking the primary physics calculation loops.

**File Structure (ASCII Tree):**
```text
mlip-pipelines/
├── src/
│   ├── api/
│   │   ├── routes.py                   # Significantly extended with the crucial /validate/run0 pre-flight diagnostic endpoint
│   │   └── **websocket.py**                # New highly concurrent, massively scalable WebSocket Connection Manager
│   ├── core/
│   │   ├── orchestrator.py
│   │   └── **telemetry_hook.py**           # New asynchronous, non-blocking metrics broadcasting hook and singleton Queue manager
│   ├── dynamics/
│   │   ├── **ff_validator.py**             # Newly implemented Hybrid force field species compatibility logic
│   │   └── dynamics_engine.py          # Heavily modified to successfully emit Run0 diagnostic data and live MD thermodynamic streaming stats
│   ├── trainers/
│   │   └── auto_hpo_manager.py         # Extensively modified to stream optimization loss curves directly to the socket
└── tests/
    └── api/
        └── test_websocket.py           # Rigorous, complex integration tests for absolute non-blocking telemetry streaming and perfect Run 0 parsing
```

The system flow mandates that the `Run 0` diagnostics occur securely via a standard, synchronous HTTP POST request directly to `/validate/run0`. The backend temporarily, safely instantiates the massive `DynamicsEngine` with a specific, hardcoded `run 0` execution override command, aggressively parsing the immediate, raw stdout output stream (strictly checking for specific LAMMPS crash codes, "Lost atoms" fatal messages, or mathematically infinite `NaN` energies), and securely returning a highly detailed, deeply nested boolean pass/fail payload populated with specific diagnostic strings and error traces. Prior to executing LAMMPS, the `ff_validator.py` intelligently intercepts the ASE `Atoms` object to cross-check element parameter mixing rules.
For the Real-time Monitoring requirement, the massive frontend application establishes a persistent, highly stateful WebSocket connection directed to `ws://.../ws/telemetry`. Throughout the continuous execution of the massive `orchestrator.run_cycle()`, sub-components like the highly complex `PacemakerTrainer` neural network optimizer or the heavily utilized `DynamicsEngine` will constantly invoke a robust global singleton via `TelemetryBroadcaster.emit()`. This incredibly sophisticated broadcaster pushes massive serialized JSON packets (containing absolutely accurate timestamps, highly specific metric names, and deep floating-point value arrays) flawlessly down the open WebSocket to absolutely all connected React GUI clients, instantly and beautifully updating their massive dashboard graphs.

## 3. Design Architecture

**Domain Concepts & Pydantic Models:**
-   **`RunZeroDiagnosticsDTO`**: A deeply robust model perfectly defined in `gui_schemas.py` that fully, comprehensively describes the immediate mathematical health of the physical structure and the script syntax.
    -   `passed`: A definitive, highly decisive Boolean.
    -   `errors`: A massive List of detailed strings explaining exactly what failed (e.g., `["Fatal atomic collision detected between IDs 5 and 12", "LAMMPS Syntax Error on line 42 regarding absolutely invalid fix arguments"]`).
    -   `missing_forcefield_params`: A strictly defined List of element symbols (e.g., `["Ni", "O"]`) that the intelligent Hybrid Force Field Validator discovered were completely missing from the user's classical potential definitions.
    -   `initial_energy`: An optional Float providing the immediate baseline thermodynamic energy of the system to verify equilibrium stability.
-   **`TelemetryPacket`**: A highly structured, meticulously optimized model elegantly representing a streamable, real-time massive data point.
    -   `timestamp`: Float (providing absolutely precise epoch time synchronization for graphing).
    -   `source`: An Enum definitively and strictly indicating the originating subsystem (`TRAINER`, `DYNAMICS`, `ORACLE`).
    -   `metric_type`: An Enum strictly defining the exact data type being transmitted (`LOSS`, `ENERGY`, `UNCERTAINTY_MAX`, `EVENT`).
    -   `value`: A highly flexible, dynamically typed Float or String payload comprehensively containing the actual data point values.

**Key Invariants, Constraints, and Validation Rules:**
1.  **Absolutely Non-Blocking Telemetry Queues**: The crucial act of continuously emitting a massive `TelemetryPacket` must absolutely never, under any conceivable circumstances, block the main computational C++ or Python physics threads. If a WebSocket client abruptly disconnects due to a closed laptop lid or terrible wifi connection, the multi-million atom Orchestrator must absolutely not crash, hang, or even slightly slow down. The `TelemetryBroadcaster` must strictly, uncompromisingly use asynchronous `asyncio.Queue` structures and extremely broad, catch-all `contextlib.suppress` wrappers to handle massive broken pipes and totally disconnected sockets with extreme grace and fault tolerance.
2.  **Strict Run 0 Environmental Isolation Security**: The `Run 0` diagnostic test must execute flawlessly in a completely isolated, mathematically ephemeral temporary directory securely provided by Python's `tempfile.TemporaryDirectory`. It must aggressively, perfectly, and successfully clean up absolutely any generated massive binaries, massive `.xyz` trajectory files, or massive `log.lammps` text files immediately after completion—regardless of whether the LAMMPS run crashed horribly or succeeded perfectly—to absolutely prevent illegally polluting the actual active project workspace and exhausting the host server's precious disk space quota limits.
3.  **Comprehensive Force Field Species Completeness**: The Hybrid Force Field Validator must meticulously extract every single unique chemical symbol natively present in the user's uploaded ASE `Atoms` array structure and mathematically compute the difference against the explicitly provided parameters. If the set difference is definitively greater than zero, the Run 0 execution must instantly abort and return the exact missing elements in the strict `missing_forcefield_params` array.

## 4. Implementation Approach

**Step 1: Concurrent, Massively Scalable WebSocket Connection Manager**
Create the entirely new `src/api/websocket.py` core module. Implement a phenomenally robust `ConnectionManager` class strictly and perfectly following FastAPI's asynchronous scaling documentation. It will flawlessly maintain a highly thread-safe list of active `WebSocket` connections and securely provide optimized `async` methods like `broadcast(packet: str)` to disseminate the massive JSON data payloads perfectly to absolutely all connected React clients simultaneously.

**Step 2: Non-Blocking Telemetry Hook and Queue Implementation**
Create the immensely important `src/core/telemetry_hook.py` module. Implement a perfectly thread-safe singleton `TelemetryBroadcaster` that exclusively utilizes Python's highly efficient `asyncio.Queue` architecture. The deeply nested, heavy core ML engines (e.g., residing within `src/trainers/`) will constantly call a very fast, entirely synchronous `emit()` method which instantly and securely places data objects directly onto the queue memory. A persistent, isolated background task definitively residing in the FastAPI application layer will continuously and asynchronously read heavily from this queue and rapidly call `ConnectionManager.broadcast()`, completely and elegantly decoupling the heavy physics math calculations from the slow network I/O operations.

**Step 3: Integrating Telemetry into Massive C++ Core Engines**
Carefully, securely modify `dynamics_engine.py` to continuously, efficiently parse the massive `log.lammps` output file periodically (or utilize highly efficient LAMMPS C++ python library native hooks if readily available in the environment) and instantly `emit()` the current MD timestep's exact thermodynamic potential energy and the foundation model's maximum atomic epistemic uncertainty array. Modify the incredibly complex ML trainer classes (like `PacemakerTrainer`) to cleanly and precisely `emit()` the exact training and validation loss calculations precisely after every single backpropagation epoch completes.

**Step 4: Intelligent Hybrid Force Field Validator and Run 0 Diagnostic Logic**
Create `src/dynamics/ff_validator.py`. Implement a function `check_forcefield_completeness(atoms: ase.Atoms, config: ProjectConfig) -> list[str]`. This function extracts all unique symbols from the atoms object using `numpy.unique` and carefully compares them against the classical potential strings explicitly defined in the active configuration tree. In `routes.py`, construct the robust `POST /validate/run0` endpoint correctly accepting a complete `ProjectConfig`. In the handler, securely call the FF Validator first. If it returns missing elements, immediately fail the validation. If it completely passes, securely extract the initial structure, elegantly generate a custom LAMMPS script specifically and deliberately replacing the standard `run N` command strictly with `run 0`, and securely execute the LAMMPS binary totally isolated in a subprocess. Aggressively, perfectly parse the standard output and standard error streams. If LAMMPS returns absolutely any non-zero exit code, or if the deeply parsed thermodynamic output natively shows the string "NaN" or wildly, impossibly positive energy values (firmly indicating extreme atomic overlap bounds), immediately populate and securely return a massively failed `RunZeroDiagnosticsDTO` without ever initiating the massive orchestrator loop.

## 5. Test Strategy

Testing this deeply interactive, massive cycle focuses incredibly intensely on verifying flawless asynchronous queue memory management, perfectly testing resilience against dropped network connections under heavy load, and beautifully parsing highly complex physics-engine crash states from massive raw text streams without ever missing an error code.

**Unit Testing Approach:**
-   **Target:** The `Run 0` massive stdout string parser.
-   **Method:** We will programmatically feed predefined, highly specific, massive string blocks flawlessly mimicking actual, terrible LAMMPS crash logs (e.g., "ERROR: Lost atoms: original 1000000 current 999999" or "Step Temp E_pair ... 0 300 NaN"). We will assert completely that the parser flawlessly and correctly flags these as total failures and accurately, cleanly extracts the relevant error message string without ever crashing on an `IndexError`.
-   **Target:** The `TelemetryBroadcaster` strict, absolutely non-blocking behavior under immense load.
-   **Method:** We will flawlessly instantiate the broadcaster singleton. We will rapidly, programmatically call `emit()` exactly 10,000 times in a tight, synchronous loop. We will definitively assert that the `asyncio` queue fills entirely appropriately without ever once blocking or slowing down the primary python thread, decisively proving it is perfectly safe for integration with high-performance physics loops.
-   **Target:** The Hybrid Force Field Validator logic.
-   **Method:** We will cleanly instantiate a mock `ase.Atoms` object containing explicitly `Pt`, `Ni`, and `O` symbols. We will pass a mocked `ProjectConfig` completely lacking parameters for `Ni`. We will assert that the validator perfectly and instantly identifies and returns exactly `["Ni"]` as the strict difference, completely catching the misconfiguration.

**Integration Testing Approach:**
-   **Target:** The absolutely critical `/ws/telemetry` WebSocket endpoint scaling.
-   **Method:** We will aggressively use `FastAPI.testclient.TestClient.websocket_connect`. We will flawlessly connect a mock dummy client. In a completely separate, strictly isolated thread, we will rapidly trigger an `emit()` call from the mock ML trainer. We will categorically assert that the test client successfully and perfectly receives the exact JSON string representation of the `TelemetryPacket` perfectly intact over the active socket without packet loss. Crucially, we will disconnect the client violently and abruptly, emit another massive packet, and assert the backend system handles the resulting broken pipe Exception beautifully, silently, and gracefully without ever crashing the massively important active background orchestrator task.
-   **Side-effect Isolation:** The massive telemetry queues will be forcefully and entirely flushed immediately after all assertions to cleanly prevent RAM memory leaks. The massive Run 0 LAMMPS execution tests will heavily and perfectly mock the core `subprocess.run` python call, explicitly and exclusively returning completely predefined `stdout` byte strings that perfectly simulate known, catastrophic LAMMPS crashes (e.g., "ERROR: Lost atoms"), absolutely preventing actual, slow binary execution and perfectly ensuring the test suite remains rapid, perfectly deterministic, and highly portable across CI environments.
