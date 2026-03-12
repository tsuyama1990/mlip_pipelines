# System Architecture for MLIP-Pipelines

## 1. Summary

The `mlip-pipelines` project aims to democratise the construction and deployment of State-of-the-Art Machine Learning Interatomic Potentials (MLIP) by automating the process around the "Pacemaker (ACE: Atomic Cluster Expansion)" tool.
This system significantly reduces the required expertise in data science and computational physics, enabling researchers to automatically traverse from initial structure generation, Density Functional Theory (DFT) calculations, active learning, up to deploying the potential in production runs.

The automated workflow mitigates the high risks of physical failure in unseen regions and the accumulation of uninformative data. By combining an Adaptive Exploration Policy with stringent physical regularisation strategies (such as Delta Learning with a Lennard-Jones baseline), the system generates physically robust potentials with significantly higher data efficiency.

## 2. System Design Objectives

### Goal
The primary objective of this project is to develop a complete, zero-configuration active learning pipeline. By only requiring a single YAML configuration file, the system must autonomously carry out exploration of structural configurations, compute exact physical properties using a DFT Oracle, train a physically-informed MLIP, and validate it using a Dynamics Engine. This zero-configuration philosophy is fundamental; users should not need to write manual Python loops or bash scripts to orchestrate the back-and-forth between DFT engines and ML trainers. The architecture must elegantly wrap these complexities into a single execution command that handles the entire lifecycle of a potential.

### Success Criteria
1.  **Zero-Config Workflow:** Automating the pipeline perfectly so users avoid writing manual Python scripts or managing loop iterations. The `config.yaml` file must act as the single source of truth for all parameters, automatically cascading logic down to the Oracle's k-point meshes and the Trainer's D-Optimality active sets.
2.  **Data Efficiency:** Utilising Active Learning and strategic structure sampling to achieve equal precision (RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å) with less than 1/10th of the DFT calculations needed in random sampling. The system must inherently filter out redundant structures, refusing to waste expensive Oracle compute time on states that offer no new mathematical information to the ACE basis set.
3.  **Physical Robustness:** Providing safety against collapsing models (due to unphysical overlapping forces) by forcing Delta Learning from a baseline (LJ/ZBL), thus preserving core-repulsion physics. The deployment of the potential into Molecular Dynamics (MD) or kinetic Monte Carlo (kMC) engines must dynamically overlay this baseline, assuring that atomic overlap results in a hard repulsive shell, eliminating Segmentation Faults entirely.
4.  **Scalability:** Assuring seamless deployment of the generated model from localised active learning to massive Molecular Dynamics (MD) or kinetic Monte Carlo (kMC) runs. The `Dynamics Engine` must support both short-timescale vibrations and long-timescale diffusion phenomena seamlessly.

### Constraints
1.  **Domain Agnosticism:** The architecture must not be strictly tied to one chemical system or one molecular structure, adapting actively to user-defined domains based on the "Material DNA" extracted during the exploration phase.
2.  **Modularity:** The Python codebase must adhere strictly to modern modularity principles. Separation of concerns is paramount to prevent any "God Classes." The `Orchestrator` handles flow control, the `Oracle` handles quantum mechanics, and the `Trainer` handles machine learning. They must not directly call one another.
3.  **Hardware Adaptability:** The system modules should be containerisable and capable of migrating between local workstations and HPC clusters without breaking. System calls to external binaries (`pw.x`, `lammps`, `pace_train`) must be wrapped in fault-tolerant exception handling that degrades gracefully or mocks outputs if the underlying binaries are missing.

## 3. System Architecture

The core of the system is the Python-based **Active Learning Orchestrator**. This component acts as the "Brain", directing the flow between four key independent modules. Strict boundaries and interfaces define how these elements interact to prevent tight coupling. The interaction relies primarily on dependency injection and robust repository patterns to exchange data, structural geometries, and model checkpoints via standard Pydantic models acting as Data Transfer Objects (DTOs).

### Core Components

*   **Structure Generator (Explorer):** Responsible for finding unseen regions in chemical/structural space. It employs an "Adaptive Exploration Policy Engine" that determines exploration parameters dynamically (e.g., Temperature schedules, MD/MC Ratio, defect densities) based on the input material's features (band gap, bulk modulus, melting point). It intelligently avoids random exploration in favour of physically meaningful deformations.
*   **Oracle (Teacher):** Wraps high-fidelity DFT simulations (e.g., Quantum ESPRESSO via ASE) to compute true forces, energies, and stress tensors. It implements critical self-healing logic for SCF convergence (automatically tweaking mixing betas or smearing widths upon failure). Crucially, it utilises "Periodic Embedding" to extract tiny, exact, periodic supercells around high-uncertainty regions, drastically reducing the DFT cost without introducing vacuum surface artifacts.
*   **Trainer (Learner):** Controls the Pacemaker tool to fit the Atomic Cluster Expansion (ACE) potentials. It employs D-Optimality based Active Set optimisation (via `pace_activeset`) to ensure only mathematically independent structures are appended to the training dataset. It strictly enforces Delta Learning against a ZBL/Lennard-Jones baseline to guarantee short-range repulsive physics.
*   **Dynamics Engine (Executor & Inference):** Employs the trained potentials in LAMMPS or EON (kMC) runs. It features a critical On-The-Fly (OTF) uncertainty quantification mechanism. By tracking the ACE extrapolation grade (`gamma`), it acts as a "watchdog", issuing a `fix halt` command the millisecond the simulation enters an extrapolated, unphysical region, returning the exact failed structure back to the Orchestrator for healing.
*   **Validator:** The Quality Assurance gate. Before deploying any `potential.yace` to the next cycle or production, it runs rigorous tests: Phonon dispersion (checking for imaginary frequencies), Mechanical Stability (Born criteria via elastic tensor calculation), and Equation of States (Birch-Murnaghan fits).

### Data Flow & Boundary Management

The Orchestrator defines a strict linear state machine: **Explore -> Detect -> Select -> Compute -> Train -> Deploy.**

1.  **Explore:** The `Dynamics Engine` runs LAMMPS with the current potential.
2.  **Detect:** The watchdog halts if `gamma > threshold`, returning a `HaltEvent` DTO to the Orchestrator.
3.  **Select:** The Orchestrator passes the failed structure to the `Structure Generator` to create local candidates (normal mode approximations), which are filtered by the `Trainer`'s D-Optimality filter.
4.  **Compute:** The filtered structures are wrapped via Periodic Embedding and sent to the `Oracle` for exact DFT evaluation.
5.  **Train:** The new DFT data is appended to the main dataset (`accumulated.pckl.gzip`), and the `Trainer` fine-tunes the potential.
6.  **Deploy:** The new potential replaces the old one, and the Orchestrator tells the `Dynamics Engine` to resume.

**Boundary Management Rules:**
*   **No Circular Dependencies:** Components must interact strictly via the Orchestrator. The `Trainer` cannot directly call the `Oracle`. If the Trainer needs new data, it asks the Orchestrator, which commands the Oracle.
*   **Stateless Execution:** The `Oracle`, `Structure Generator`, and `Validator` must be completely stateless. They accept inputs (structures, parameters) and return outputs (forces, scores, modified structures). All persistent state (datasets, current potential path, iteration count) is managed exclusively by the Orchestrator and written to designated file repositories.

### Component Diagram

```mermaid
graph TD
    A[Active Learning Orchestrator] -->|ExplorationStrategy| B(Structure Generator)
    A -->|HaltEvent Geometries| C(DFT Oracle)
    A -->|Accumulated Dataset| D(Trainer)
    A -->|potential.yace & hybrid pairs| E(Dynamics Engine)
    A -->|Validation Requests| F(Validator)

    B -->|Defected Candidates| A
    C -->|Ground Truth Energies/Forces| A
    D -->|potential.yace| A
    E -->|HaltEvent (gamma > threshold)| A
    F -->|ValidationScore (Phonon, RMSE)| A
```

## 4. Design Architecture

This project adopts an "Additive Mindset." It will reuse the existing standard Python project layout initialized by `uv` and safely extend the domain models and interfaces. The core architecture relies heavily on `pydantic` for schema validation and strict type hints (`mypy`) to enforce API contracts between the independent modules.

### Additive Mindset: Existing vs. New Files

**Existing Assets Reused:**
*   `pyproject.toml`: Retained and appended with strict linting (`ruff`) and testing (`pytest`, `mypy`) rules.
*   `README.md`: Completely rewritten to act as the primary landing page, reflecting the new capabilities.
*   `main.py`: Modified solely to import the new `ActiveLearningOrchestrator` and parse CLI arguments, keeping the entry point clean.
*   `dev_documents/ALL_SPEC.md`: Treated as the immutable source of truth.

**New Assets to be Created (Safely Extended):**
The new logic will be encapsulated entirely within the `src/` directory, following a modular domain-driven design. This ensures that the new Active Learning loops do not interfere with any basic scripting structures that might exist, isolating the complexity into well-defined packages.

### File Structure Overview

```ascii
project_root/
├── src/
│   ├── core/
│   │   ├── orchestrator.py      # ActiveLearningOrchestrator class
│   │   ├── config_schemas.py    # Pydantic Core Models (PipelineConfig)
│   │   └── exceptions.py        # Domain Exceptions (HaltEventInterrupt)
│   ├── generators/
│   │   ├── adaptive_policy.py   # Adaptive Exploration Policy Engine
│   │   └── defect_builder.py    # Atomic manipulators (Normal mode, rattling)
│   ├── oracles/
│   │   ├── qe_manager.py        # Quantum Espresso Interface & Self-Healing
│   │   └── embedder.py          # Periodic Embedding logic for Halt structures
│   ├── trainers/
│   │   ├── pacemaker_wrapper.py # Trainer integration & Delta Learning setup
│   │   └── active_set.py        # D-Optimality Filter (MaxVol algorithms)
│   ├── dynamics/
│   │   ├── lammps_interface.py  # Hybrid Potential setup & Watchdog
│   │   └── eon_wrapper.py       # kMC integration (Dimer searches)
│   └── validators/
│       ├── stability_tests.py   # Phonon (Phonopy) & Elasticity (Born) tests
│       └── reporter.py          # HTML Report generation
├── tests/
│   ├── unit/                    # Fast, mocked tests per module
│   ├── integration/             # Component interaction tests
│   └── e2e/                     # Full pipeline dummy runs
├── pyproject.toml
└── README.md
```

### Core Domain Pydantic Models Structure

The backbone of the system's modularity lies in its Pydantic models (`src/core/config_schemas.py`). These models validate the `config.yaml` and serve as rigid DTOs between the Orchestrator and its modules.

1.  **`PipelineConfig` (Root DTO):**
    *   Parses the entire user YAML. Contains nested sub-models: `SystemConfig` (elements, mass), `OracleConfig` (k-spacing targets, pseudo paths), `TrainerConfig` (ACE max degree, LJ baseline parameters), and `DynamicsConfig` (gamma thresholds).

2.  **`ExplorationStrategy` (Generator -> Orchestrator DTO):**
    *   Outputted by the Adaptive Policy Engine. Defines `r_md_mc` (float), `t_schedule` (list of floats), `n_defects` (int), and `strain_range` (float). The Orchestrator passes this to the Dynamics Engine to configure the run.

3.  **`HaltEvent` (Dynamics -> Orchestrator DTO):**
    *   Raised as a custom Python exception or returned cleanly. Contains `timestep` (int), `max_gamma` (float), `triggering_atoms_indices` (list[int]), and `halt_structure` (ASE Atoms object).

4.  **`ValidationScore` (Validator -> Orchestrator DTO):**
    *   Contains `rmse_energy` (float), `rmse_force` (float), `phonon_stable` (bool), and `born_criteria_passed` (bool). Defines if the Orchestrator promotes the potential to the next iteration.

By strictly enforcing these Pydantic models, the Python type checker (`mypy`) guarantees that the `Oracle` will never receive a malformed configuration dictionary, and the `DynamicsEngine` will always return exactly the data required to heal the potential.

## 5. Implementation Plan

The development is divided strictly into 8 sequential implementation cycles. Each cycle must be fully implemented, tested, and validated before proceeding to the next, ensuring a robust, layered architecture.

### Cycle 01: Core Framework and Pydantic Schemas
This foundational cycle establishes the rigid boundaries of the system.
*   **Tasks:**
    1.  Initialize the `src/core/` directory.
    2.  Implement `config_schemas.py` using `pydantic`. Map every single parameter defined in the `ALL_SPEC.md` (e.g., mixing betas, k-spacing, ACE cuts, gamma thresholds) into strongly typed, nested Pydantic models (`PipelineConfig`, `SystemConfig`, etc.) with sensible default values.
    3.  Implement `exceptions.py` to define the domain-specific errors (e.g., `OracleConvergenceError`, `DynamicsHaltInterrupt`).
    4.  Create the abstract base classes (ABCs) or Protocols for the four main modules (`AbstractOracle`, `AbstractTrainer`, `AbstractDynamics`, `AbstractGenerator`). This enforces the Dependency Injection contract for the Orchestrator.
*   **Integration Points:** The `PipelineConfig` model will be imported by `main.py` to parse the CLI YAML file and validate it instantly before any heavy modules are loaded.

### Cycle 02: Adaptive Exploration Policy Engine
This cycle builds the "Explorer" intelligence, moving away from hardcoded configurations.
*   **Tasks:**
    1.  Implement `src/generators/adaptive_policy.py`. This engine will take generic material descriptors (e.g., band gap = 0 for metals, complex unit cell for insulators) and output an `ExplorationStrategy` Pydantic model.
    2.  Write the decision logic matrix (e.g., if metal -> High-MC Policy; if insulator -> Defect-Driven Policy).
    3.  Implement `defect_builder.py` utilizing `ase.build` and `pymatgen` to physically insert vacancies, anti-site defects, or apply volumetric strain based on the policy's output.
*   **Integration Points:** The Orchestrator will call the `AdaptivePolicy` at the start of an iteration to dictate exactly how the `DynamicsEngine` should format its LAMMPS input script.

### Cycle 03: The DFT Oracle and Periodic Embedding
This cycle handles the quantum mechanical heavy lifting and the critical self-healing logic.
*   **Tasks:**
    1.  Implement `src/oracles/qe_manager.py` wrapping `ase.calculators.espresso.Espresso`.
    2.  Build the dynamic parameter generation: translate `kspacing` into actual k-point grids based on cell volume; auto-select SSSP pseudopotentials based on atomic numbers; auto-enable spin-polarization for magnetic elements (Fe, Co, Ni).
    3.  Implement the Self-Correction loop: catch `scf_convergence` failures, automatically lower `mixing_beta`, increase smearing, and retry up to 3 times.
    4.  Implement `src/oracles/embedder.py`. This mathematical logic takes a `HaltEvent` structure, isolates the high-gamma atoms, cuts an Orthorhombic Box of size $R_{cut} + R_{buffer}$, and applies periodic boundary conditions without vacuum gaps.
*   **Integration Points:** The Orchestrator will pass raw or embedded ASE Atoms objects to the Oracle. The Oracle returns the same objects but with `.get_forces()` and `.get_potential_energy()` populated, ready for the Trainer.

### Cycle 04: Trainer and Pacemaker Integration
This cycle encapsulates the Machine Learning generation and data curation.
*   **Tasks:**
    1.  Implement `src/trainers/pacemaker_wrapper.py`. This class uses `subprocess.run` to execute the `pace_train` CLI tool in a temporary directory, passing the correctly formatted YAML configurations.
    2.  Implement the Delta Learning logic. The wrapper must automatically generate the configuration lines to subtract a ZBL or Lennard-Jones baseline from the target DFT energies before fitting the ACE polynomials.
    3.  Implement `src/trainers/active_set.py`. Wrap the `pace_activeset` command to filter large batches of candidate structures down to the 5-10 most mathematically informative points (maximizing D-Optimality) before appending them to the main dataset.
*   **Integration Points:** The Trainer consumes the force-populated structures from the Oracle, updates the central `accumulated.pckl.gzip`, runs the training, and returns the absolute path to the newly minted `potential.yace` file.

### Cycle 05: Dynamics Engine and On-The-Fly (OTF) Handling
This cycle builds the operational execution engine that runs the potentials and acts as the watchdog.
*   **Tasks:**
    1.  Implement `src/dynamics/lammps_interface.py`. This uses the Python `lammps` module or robust subprocesses to execute MD runs.
    2.  Crucially, auto-generate the `in.lammps` script to explicitly require `pair_style hybrid/overlay pace zbl`. This guarantees the Delta Learning baseline is physically enforced during the simulation.
    3.  Implement the Watchdog script block: inject `compute pace_gamma` and `fix halt` commands into LAMMPS, triggering an immediate stop if `v_max_gamma > threshold`.
    4.  Parse the LAMMPS log/dump file upon a halt to extract the exact timestep and coordinates of the failing state, wrapping it into a `HaltEvent` Pydantic model.
*   **Integration Points:** The Dynamics Engine receives the `potential.yace` path from the Orchestrator, runs until completion or an OTF interrupt, and returns either a success flag or the `HaltEvent` back up the chain.

### Cycle 06: Active Learning Loop Closure (Orchestrator)
This cycle wires the previously built independent modules into the autonomous loop.
*   **Tasks:**
    1.  Implement the main `ActiveLearningOrchestrator.run_cycle()` method in `src/core/orchestrator.py`.
    2.  Code the specific Halt-and-Diagnose sequence: Receive `HaltEvent` -> Call Generator for local candidates (Normal Mode Approximation) -> Call Trainer to filter candidates via D-optimality -> Call Oracle's Embedder to wrap the candidates -> Call Oracle to compute DFT forces -> Call Trainer to fine-tune the potential -> Tell Dynamics to resume.
    3.  Implement directory state management, ensuring every iteration (e.g., `active_learning/iter_001/`) creates isolated logs, DFT runs, and local candidate structures without overwriting previous data.
*   **Integration Points:** This is the central hub. It injects the `PipelineConfig` into all modules, manages the while-loop, and handles all overarching exception logging.

### Cycle 07: kMC Extension and Advanced Validation
This cycle scales the time domain and enforces quality assurance.
*   **Tasks:**
    1.  Implement `src/dynamics/eon_wrapper.py` to support Adaptive Kinetic Monte Carlo via EON. Ensure the `pace_driver.py` script correctly injects the ACE potential into the Dimer method saddle-point searches, triggering the same OTF halts if `gamma` spikes during transition states.
    2.  Implement `src/validators/stability_tests.py`. Integrate `Phonopy` via ASE to calculate finite displacement force constants and evaluate the Brillouin zone for imaginary frequencies.
    3.  Implement the Elasticity checking (Born criteria) by applying $\pm 1\%$ and $\pm 2\%$ strains to the unit cell and fitting the $C_{ij}$ tensor.
    4.  Implement `src/validators/reporter.py` to generate the HTML validation report containing Parity Plots and Phonon bands.
*   **Integration Points:** The Orchestrator calls the Validator after the Trainer completes a major cycle, evaluating the `ValidationScore` to decide whether to deploy the potential or request more data.

### Cycle 08: End-to-End Orchestration & User Documentation
The final polish, focusing on user experience, tutorials, and CLI robustness.
*   **Tasks:**
    1.  Finalize `src/main.py`, building a robust `argparse` or `click` CLI interface that seamlessly loads the YAML and initiates the `Orchestrator`.
    2.  Implement robust Python `logging` to capture the entire pipeline's progress, funneling Oracle SCF outputs, LAMMPS thermodynamic prints, and Trainer epochs into a unified `report.json` and terminal output.
    3.  Develop the comprehensive `tutorials/UAT_AND_TUTORIAL.py` using Marimo, ensuring the mock functionality perfectly mimics the state machine transitions so users can experience the pipeline in seconds.
*   **Integration Points:** The CI/CD pipeline connects to the Marimo tests, proving the entire zero-config workflow is operational end-to-end.

## 6. Test Strategy

Due to the extreme computational cost of DFT and ML training, the test strategy strictly enforces pure unit testing via extensive mocking, utilizing `unittest.mock` and `tempfile` to prevent any side effects or actual system calls during CI.

### Cycle 01: Core Framework
*   **Unit Tests:** Validate `PipelineConfig` using `pytest`. Provide intentionally malformed YAML dictionaries (e.g., missing required elements, strings instead of ints for k-spacing) and assert that Pydantic raises the correct `ValidationError`.
*   **Side-effect Prevention:** Pure Python memory tests; no file I/O necessary beyond reading dummy YAML strings.

### Cycle 02: Exploration Engine
*   **Unit Tests:** Test the `AdaptivePolicy` decision matrix. Input a mock "Metal" descriptor and assert the output `ExplorationStrategy` correctly specifies high MD/MC ratios. Input an "Insulator" descriptor and assert defect densities are elevated.
*   **Integration Tests:** Test the `defect_builder.py` using basic ASE Aluminum FCC structures to assert vacancies are correctly removed without invoking LAMMPS.

### Cycle 03: DFT Oracle
*   **Unit Tests (Embedder):** Pass a mock 1000-atom ASE supercell with a designated "high gamma" centre. Assert that the `Periodic Embedding` function correctly returns a small $\sim$50 atom Orthorhombic cell that perfectly satisfies PBC geometry.
*   **Mocking:** To test `qe_manager.py`, patch `ase.calculators.espresso.Espresso.get_potential_energy` using `unittest.mock.patch`. The test must assert that the SCF retry loop correctly triggers (e.g., `mixing_beta` lowers) when the mock throws a `ConvergenceError`, without ever running the `pw.x` binary.
*   **Side-effect Prevention:** All file I/O (pseudo downloads, `.pwi` generations) must be strictly confined within a `tempfile.TemporaryDirectory` block.

### Cycle 04: Trainer Integration
*   **Unit Tests:** Test the configuration generator to ensure the hybrid ZBL/Lennard-Jones baseline strings are correctly formatted into the Pacemaker YAML inputs.
*   **Mocking:** Patch `subprocess.run` to intercept calls to `pace_train` and `pace_activeset`. For active set testing, pass a dummy list of 100 structures and mock the subprocess to return exactly 5 specific indices, verifying the `ActiveSet` class slices the dataset correctly.
*   **Side-effect Prevention:** Mock the creation of the `potential.yace` file using `pathlib.Path.touch()` within a temporary directory to simulate a successful training run.

### Cycle 05: Dynamics Engine
*   **Unit Tests:** Assert that `lammps_interface.py` accurately generates the `in.lammps` template strings, specifically verifying the presence of the `pair_style hybrid/overlay pace zbl` and `fix halt` lines.
*   **Integration Tests:** Simulate a `HaltEvent`. Provide a mock LAMMPS dump file containing a high `gamma` reading. Assert the parser correctly extracts the timestep, the offending atom IDs, and reconstructs the specific ASE Atoms object.
*   **Side-effect Prevention:** No actual LAMMPS binaries will be executed. The `run_exploration` method will be patched to instantly return a pre-constructed `HaltEvent` DTO.

### Cycle 06: Loop Closure (Orchestrator)
*   **Integration Tests:** This is the critical state-machine test. Mock all four modules (`Explorer`, `Oracle`, `Trainer`, `Dynamics`). Initiate `Orchestrator.run_cycle()`.
    *   Setup the mock `Dynamics` to return a `HaltEvent`.
    *   Assert `Orchestrator` calls `Oracle.compute()` with the specific embedded structure.
    *   Assert `Orchestrator` calls `Trainer.train()` with the new data.
    *   Assert `Orchestrator` calls `Dynamics.resume()`.
*   **Side-effect Prevention:** The test purely verifies the call stack and argument passing between the mock objects, taking milliseconds instead of days.

### Cycle 07: KMC and Validation
*   **Unit Tests (Validator):** Test the Mechanical Stability logic. Provide a hardcoded mathematical array representing an elastic $C_{ij}$ tensor that violates the Born criteria (e.g., $C_{44} < 0$). Assert the `Validator` returns `born_criteria_passed = False`.
*   **Mocking:** Patch `Phonopy` execution. Have the mock return a dummy band structure with imaginary frequencies ($\omega^2 < 0$) and assert the `Validator` accurately flags it as unstable.

### Cycle 08: End-to-End E2E Testing
*   **E2E Tests:** Execute the Marimo tutorial notebook (`tutorials/UAT_AND_TUTORIAL.py`) via the CI pipeline in headless mode with the `USE_MOCK = True` flag.
*   **Side-effect Prevention:** The tutorial itself contains the dependency injection hooks to utilize dummy Oracles and Trainers. The test ensures that the full sequence, from initial CLI invocation to final energy property calculation (FePt/MgO interface energy), executes flawlessly, proving the Zero-Config pipeline is mathematically and logically sound.