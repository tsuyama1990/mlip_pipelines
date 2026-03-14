# User Test Scenario & Tutorial Plan

## 1. Test Scenarios

### Scenario 1: FePt/MgO Interface Energy and Transitions (The Aha! Moment)
**ID:** UAT-001
**Priority:** High

**Description:**
This scenario serves as the defining "Aha! Moment" for the user. They will utilize the newly built MLIP pipeline to seamlessly compute complex properties of the FePt/MgO system. The user wants to evaluate the interface energy between FePt and MgO, understand shear transitions, calculate the FePt order parameter, and examine specific end-faces (Fe-face vs Pt-face) in contact with MgO.

Typically, achieving this requires a sophisticated understanding of both data science (to build an active learning potential from scratch) and computational physics (to meticulously set up Quantum ESPRESSO DFT runs and LAMMPS relaxations for interface boundaries). Using our automated zero-configuration pipeline, the user simply supplies the YAML file and watches as the system autonomously explores the chemical space, samples specific defects and configurations, generates exact forces/energies using the Oracle (with periodic embedding to isolate interface features correctly), and iteratively trains an ACE model. Finally, the user invokes the trained potential inside a provided Marimo tutorial script to calculate the required interface energies, shear energies, and order parameters automatically. The result amazes the user because tasks that used to take weeks of iterative manual labour and debugging are completed autonomously in a matter of hours.

### Scenario 2: Active Learning "Halt and Heal" Verification
**ID:** UAT-002
**Priority:** Medium

**Description:**
This scenario tests the critical "On-The-Fly" (OTF) failure detection and self-healing mechanism. A user subjects a simple crystalline system (e.g., pure Titanium) to an extreme high-temperature MD run that forces the atoms into an unexplored configuration (extrapolation region). The user will watch as the Dynamics Engine monitors the extrapolation grade ($\gamma$), detects the high uncertainty, cleanly halts the MD run without a segmentation fault (thanks to the baseline LJ regularisation), extracts the exact uncertain local environments, runs DFT only on those small embedded periodic cells, refines the potential, and seamlessly resumes the MD run. The user experiences an "Aha! Moment" by observing the system intelligently "heal" itself, saving them from the typical frustrating cycle of manual crash-debugging and dataset expansion.

## 2. Behavior Definitions

**Feature:** FePt/MgO Interface Energy Computation
**Scenario:** End-to-end interface energy pipeline execution

**GIVEN** the user has a local installation of the mlip-pipelines system
**AND** the user has prepared a base `config.yaml` specifying Fe, Pt, Mg, and O elements
**AND** the system is configured to target interface boundary structures
**WHEN** the user initiates the Active Learning Orchestrator via the CLI
**THEN** the Structure Generator will automatically propose FePt/MgO interface geometries with both Fe and Pt terminating faces
**AND** the DFT Oracle will perform self-healing static calculations on the geometries using SSSP libraries
**AND** the Trainer will automatically fit an ACE potential utilizing D-Optimality to minimize dataset size
**AND** the Validator will ensure the potential meets RMSE Energy constraints < 1 meV/atom
**AND** the user can execute a final inference script yielding the interface energy and order parameter directly.

**Feature:** On-The-Fly (OTF) Uncertainty Healing
**Scenario:** Safe halting and iterative learning during unstable dynamics

**GIVEN** an active MD simulation governed by a newly trained ACE potential
**AND** the temperature schedule is ramped rapidly to induce unphysical structural overlaps
**WHEN** the atomic configurations yield a maximum extrapolation grade ($\gamma$) exceeding the configured threshold (e.g. 5.0)
**THEN** the Dynamics Engine will safely halt execution before atomic overlap crashes the LAMMPS engine
**AND** the Orchestrator will extract the exact structural geometry responsible for the uncertainty
**AND** the Orchestrator will apply Periodic Embedding to generate valid, small-scale DFT calculation cells
**AND** the Oracle will compute new ground truth forces for these cells
**AND** the Trainer will run fine-tuning on the ACE potential
**AND** the MD simulation will automatically resume from the last known stable state with the newly updated potential.

## 3. Tutorial Strategy

To transform the UAT into a fully interactive and convincing experience, we will provide a unified interactive tutorial using `marimo`.

**Mock Mode vs Real Mode:**
*   **Mock Mode (CI / Demo Execution):** The tutorial will use `unittest.mock` components for the `DFTManager` and `MDInterface` by default. This allows the user to click through the entire cycle (exploration, OTF halt, selection, dataset update, and resumption) in seconds on any standard laptop without needing a powerful HPC cluster or active Quantum ESPRESSO installation. It proves the logic and the state transitions of the Orchestrator.
*   **Real Mode:** The tutorial provides clear flags (e.g., `USE_MOCK = False`) which, when toggled on a properly configured machine, will invoke actual LAMMPS and Pacemaker commands to compute genuine FePt/MgO geometries.

The strategy focuses on visual feedback: plotting energy convergence, tracking dataset size growth, and visualizing the `gamma` spikes during OTF events.

## 4. Tutorial Plan

All scenarios, including the Quick Start dummy run and the Advanced "Aha! Moment" (FePt/MgO computation), will be consolidated into a **SINGLE** Marimo Text/Python file.

**File Location:** `tutorials/UAT_AND_TUTORIAL.py`

**Structure of `tutorials/UAT_AND_TUTORIAL.py`:**
1.  **Introduction & Setup:** Importing the `mlip-pipelines` library and configuring the environment (Mock vs Real).
2.  **Phase 1: The Zero-Config Run:** Running a basic single-cycle iteration (UAT-002) to showcase the "Halt and Heal" loop visually. Marimo will plot the $\gamma$ values triggering the halt.
3.  **Phase 2: The Aha! Moment:** Configuring the system for FePt/MgO interface structures. Calling the orchestrator to resolve the interface boundary and calculating the exact interface energies using the generated potential.
4.  **Phase 3: Validation:** Showing the Validator's HTML output directly inside the Marimo notebook, confirming Phonon and mechanical stability.

## 5. Tutorial Validation

To validate the tutorial, the CI pipeline will execute the Marimo script in headless mode (using `marimo run tutorials/UAT_AND_TUTORIAL.py`) with `USE_MOCK = True`.
The validation process guarantees that:
*   No Python exceptions occur during execution.
*   The orchestrator correctly loops through at least one OTF event.
*   The final output dictionary correctly reports the calculated mock energies (e.g., FePt/MgO Interface Energy and Order Parameter).
*   The final `potential.yace` file is demonstrably written to the `potentials/` directory.