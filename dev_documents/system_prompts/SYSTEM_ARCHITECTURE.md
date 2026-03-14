# System Architecture

## 1. Summary

The mlip-pipelines system is a comprehensive, fully automated framework designed to construct and refine Machine Learning Interatomic Potentials (MLIPs) using an Active Learning methodology. By tightly integrating active Molecular Dynamics (MD) and kinetic Monte Carlo (kMC) engines with automated Density Functional Theory (DFT) calculations and Advanced Configuration Exploration (ACE) model training, the framework autonomously explores complex chemical spaces, identifies configurations with high prediction uncertainty on-the-fly, computes ground-truth ab initio forces for these novel structures, and continuously fine-tunes the interatomic potential. This cyclical process dramatically accelerates the development of robust, highly accurate ML potentials, eliminating the traditional, tedious manual intervention required for dataset generation and model refinement, thus enabling rapid, high-fidelity materials discovery and property prediction.

## 2. System Design Objectives

The core objective of the mlip-pipelines architecture is to deliver a robust, highly autonomous, zero-configuration active learning pipeline capable of discovering and learning complex atomic interactions.

**Goals:**
*   **Complete Automation:** The system must execute the entire active learning loop (Exploration $\rightarrow$ Halt $\rightarrow$ Candidate Selection $\rightarrow$ DFT Evaluation $\rightarrow$ Potential Training $\rightarrow$ Resumption) without any human intervention. The user provides a single YAML configuration file, and the system handles the rest.
*   **On-The-Fly (OTF) Self-Healing:** The system must actively monitor the extrapolation grade ($\gamma$) during dynamic simulations (MD or kMC). Upon detecting high uncertainty indicating an unexplored region of phase space, it must safely halt the dynamics, extract the problematic structures, compute accurate DFT forces, retrain the model, and seamlessly resume the simulation.
*   **High Scalability & Performance:** The architecture must leverage robust, production-ready engines (LAMMPS, EON, Quantum ESPRESSO via ASE, and Pacemaker) while ensuring the orchestrating Python layer remains lightweight and resilient to long-running, compute-intensive subprocesses.
*   **Modular Extensibility:** The system must be designed using rigorous separation of concerns. The central Orchestrator must orchestrate independent, interchangeable modules (Oracles, Trainers, Dynamics Engines). This modularity allows for the future integration of alternative DFT codes or ML potential architectures without requiring a complete system rewrite.
*   **Strict Security & Reproducibility:** The pipeline must enforce strict directory isolation, sandboxing all external binary executions to prevent side-effects or environment corruption. Configuration models must be strictly validated to prevent path traversal attacks or injection of malicious parameters, ensuring every experiment is mathematically reproducible and computationally secure.

**Constraints:**
*   The system must strictly adhere to a "Hub and Spoke" Dependency Injection model, where the Orchestrator acts as the sole state manager, preventing circular dependencies between functional modules.
*   All data transfer between components must be serialized through explicit, strictly typed Pydantic models (Data Transfer Objects).
*   The core business logic must operate cleanly using temporary directories for all intermediate calculations, guaranteeing atomic commits of generated potentials and datasets to the main project directories.
*   Memory consumption must be strictly bounded, particularly when processing large datasets or reading massive potential files, necessitating streaming reads and explicit chunking.

**Success Criteria:**
*   The system successfully completes a full autonomous cycle of the Active Learning loop without manual intervention, specifically demonstrating the ability to detect an OTF event, refine the dataset, and resume the halted simulation.
*   The architecture seamlessly integrates the new Interface Generation features (e.g., FePt/MgO) alongside existing MD and kMC exploration strategies, proving the extensibility of the domain models and the Structure Generator.
*   All architectural boundaries are maintained, meaning no module (e.g., Trainer) calls another module (e.g., Oracle) directly; all flow is managed strictly by the Orchestrator.
*   The system passes a rigorous test suite demonstrating resilience against malicious configurations, file I/O errors, and expected subprocess failures.

## 3. System Architecture

The architecture employs a modern, strictly modular design centered around an `ActiveLearningOrchestrator`. This approach ensures strict boundary management and separation of concerns.

### 3.1 Components and Boundary Management

*   **Orchestrator (The Hub):** The central state machine. It is the only component that maintains the state of the active learning iteration, manages directory structures, and coordinates the sequential execution of the other modules.
*   **Dynamics Engine (Explorer):** Responsible for exploring the phase space using MD (LAMMPS) or kMC (EON). It operates strictly on the provided potential and input structure, monitoring uncertainty and returning a `HaltInfo` state to the Orchestrator. It has no knowledge of the Trainer or the Oracle.
*   **Structure Generator:** Proposes new configurations, including the generation of specific interface structures (e.g., FePt/MgO) or local perturbations around uncertain geometries.
*   **Oracle (DFT Manager):** Strictly responsible for evaluating a batch of provided structures and returning ground-truth energies, forces, and stresses. It utilizes ASE and external DFT codes (Quantum ESPRESSO) and knows nothing about the broader active learning context.
*   **Trainer (ACE Trainer):** Manages the dataset and executes the machine learning training process (Pacemaker) to produce a new `.yace` potential file. It operates purely on the provided dataset and initial potential, unaware of how the data was generated.
*   **Validator:** Assesses the quality and physical validity of the newly trained potential, acting as a quality assurance gate before the Orchestrator deploys the potential for the next iteration.

### 3.2 Explicit Rules on Separation of Concerns

1.  **Zero Circular Dependencies:** No module outside the Orchestrator may hold a reference to the Orchestrator or any other peer module.
2.  **Stateless Modules:** The Dynamics Engine, Oracle, Trainer, and Structure Generator must be completely stateless between method invocations. All required state (paths, configurations) must be explicitly passed by the Orchestrator via Pydantic DTOs or primitive arguments.
3.  **Atomic Operations:** The Orchestrator must execute all module logic within isolated temporary directories. Only upon complete, successful validation of an iteration's outputs are the artifacts atomically moved to the persistent project directories (`active_learning`, `potentials`, `data`).
4.  **Configuration Immutability:** The Pydantic configuration models (`SystemConfig`, `OracleConfig`, etc.) are instantiated once at startup and passed down immutably. Modules must not modify their configuration state during execution.

### 3.3 Data Flow and Interactions

```mermaid
graph TD
    User([User Config]) --> ConfigModel[Strict Pydantic Config]
    ConfigModel --> Orchestrator

    subgraph Active Learning Loop
        Orchestrator --> |1. Run Exploration| DynamicsEngine[Dynamics Engine\nMD/kMC]
        DynamicsEngine --> |2. HaltInfo \n(Uncertainty Detected)| Orchestrator
        Orchestrator --> |3. Generate Candidates| StructureGenerator[Structure Generator]
        StructureGenerator --> |4. Perturbed Structures| Orchestrator
        Orchestrator --> |5. Compute Ground Truth| Oracle[DFT Oracle]
        Oracle --> |6. Evaluated Structures| Orchestrator
        Orchestrator --> |7. Update Dataset & Train| Trainer[ACE Trainer]
        Trainer --> |8. New Potential| Orchestrator
        Orchestrator --> |9. Validate Potential| Validator[Quality Validator]
        Validator --> |10. Validation Result| Orchestrator
    end

    Orchestrator --> |11. Deploy Potential| FileSystem[(Persistent Storage)]
    FileSystem --> |12. Resume Run| DynamicsEngine
```

## 4. Design Architecture

The project is structured to enforce the defined architectural boundaries, leveraging standard Python directory conventions.

### 4.1 File Structure

```text
mlip-pipelines/
├── src/
│   ├── core/
│   │   ├── __init__.py          # Abstract Base Classes defining module contracts
│   │   ├── exceptions.py        # Domain-specific exceptions (e.g., DynamicsHaltInterrupt)
│   │   └── orchestrator.py      # The ActiveLearningOrchestrator implementation
│   ├── domain_models/
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic configuration schemas
│   │   └── dtos.py              # Pydantic Data Transfer Objects (HaltInfo, ValidationReport)
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── dynamics_engine.py   # LAMMPS MD Interface implementation
│   │   ├── eon_wrapper.py       # EON kMC Interface implementation
│   │   └── security_utils.py    # Path and environment security validation logic
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── adaptive_policy.py   # Strategy selection for exploration
│   │   ├── defect_builder.py    # Specific defect generation logic
│   │   └── structure_generator.py # AbstractGenerator implementation, including interface generation
│   ├── oracles/
│   │   ├── __init__.py
│   │   └── dft_oracle.py        # AbstractOracle implementation using Quantum ESPRESSO
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── ace_trainer.py       # AbstractTrainer implementation wrapping Pacemaker
│   └── validators/
│       ├── __init__.py
│       ├── reporter.py          # HTML report generation
│       ├── stability_tests.py   # Physical stability calculations
│       └── validator.py         # Main quality assurance logic
├── dev_documents/
│   ├── system_prompts/
│   │   └── SYSTEM_ARCHITECTURE.md   # This document
│   └── USER_TEST_SCENARIO.md    # User Acceptance Tests and Tutorial Plan
├── tests/                       # Comprehensive test suite (unit, integration, e2e)
├── tutorials/                   # User-facing Marimo notebooks
│   └── UAT_AND_TUTORIAL.py      # Interactive tutorial demonstrating the pipeline
├── pyproject.toml               # Project configuration and linter settings
└── README.md                    # Project landing page
```

### 4.2 Core Domain Pydantic Models Structure and Typing

The core configuration is defined in `src/domain_models/config.py` using Pydantic v2 to guarantee strict type safety and data validation.

*   `ProjectConfig`: The root settings model loaded from `.env` and YAML, coordinating the sub-configurations. It implements strict validation against path traversal and ensures the execution environment is secure.
*   `SystemConfig`: Defines the physical system, including elements and the new `InterfaceTarget` schema.
*   `DynamicsConfig`: Configuration for LAMMPS/EON exploration, defining thresholds and step limits.
*   `OracleConfig`: Defines parameters for Quantum ESPRESSO (k-spacing, cutoffs, max retries).
*   `TrainerConfig`: Configuration for Pacemaker (epochs, max potential size, active set size).
*   `StructureGeneratorConfig`: Defines constraints for structure generation, including whitelists for valid interface targets.

**Integration Points and Additive Extensions:**
The architecture cleanly extends the existing domain objects to support the new requirements specified in `ALL_SPEC.md` without disrupting established workflows:
*   The `SystemConfig` object has been additively extended to include an optional `interface_target` field, utilizing the new `InterfaceTarget` schema. This allows users to specify complex multi-material boundaries (e.g., FePt/MgO) alongside simple bulk systems.
*   The `Orchestrator` detects the presence of the `interface_target` in the `SystemConfig` during iteration 0 and seamlessly triggers the `StructureGenerator` to pre-generate the starting geometry, inserting it into the pipeline before standard exploration begins.
*   The `HaltInfo` DTO has been refined to explicitly handle high-gamma environments extracted during an OTF halt, ensuring standard formatting between the Dynamics Engine and the Structure Generator.

## 5. Implementation Plan

The project development is decomposed into six strictly sequential cycles, ensuring a stable, additive build process.

*   **Cycle 01: Core Architecture & Configuration Infrastructure**
    *   **Focus:** Establish the foundational directory structure, Abstract Base Classes (`src/core/__init__.py`), and the comprehensive Pydantic configuration models (`src/domain_models/config.py`).
    *   **Implementation:** Define all configuration schemas (`ProjectConfig`, `SystemConfig`, `InterfaceTarget`, etc.) with strict validation rules (path traversal prevention, whitelist checking). Define the core DTOs (`HaltInfo`, `ExplorationStrategy`) in `dtos.py`. Implement robust security utility functions in `security_utils.py` to handle safe environment loading and path resolution.
    *   **Output:** The fully typed configuration layer capable of safely parsing user inputs and environment variables, ready to be injected into the functional modules.

*   **Cycle 02: Dynamics Engines (Exploration)**
    *   **Focus:** Implement the `MDInterface` (LAMMPS) and `EONWrapper` (kMC) fulfilling the `AbstractDynamics` contract.
    *   **Implementation:** Build `src/dynamics/dynamics_engine.py` to orchestrate LAMMPS subprocesses. Implement the OTF halt detection logic by monitoring the `pace_gamma` compute. Build `src/dynamics/eon_wrapper.py` for kMC exploration. Ensure both engines return the strictly typed `HaltInfo` DTO when a high-uncertainty event triggers an interrupt.
    *   **Output:** The exploration layer capable of simulating atomic systems and safely halting upon detecting extrapolation regions.

*   **Cycle 03: Structure Generation & Interface Building**
    *   **Focus:** Implement the `StructureGenerator` fulfilling the `AbstractGenerator` contract, and the `AdaptiveExplorationPolicyEngine`.
    *   **Implementation:** Build `src/generators/structure_generator.py` to handle standard defect generation and local candidate perturbations around halted structures. Critically, implement the `generate_interface` method to handle the additive `InterfaceTarget` requirement (e.g., building FePt/MgO boundaries using ASE `bulk` and `stack` functions).
    *   **Output:** The generation layer capable of proposing new, targeted structural configurations for the Oracle to evaluate.

*   **Cycle 04: The DFT Oracle (Ground Truth Evaluation)**
    *   **Focus:** Implement the `DFTManager` fulfilling the `AbstractOracle` contract.
    *   **Implementation:** Build `src/oracles/dft_oracle.py` to orchestrate Quantum ESPRESSO calculations via ASE. Implement the critical "Periodic Embedding" logic to properly isolate and evaluate the local high-uncertainty environments extracted during an OTF halt. Ensure robust error handling and retry mechanisms for SCF convergence failures (`OracleConvergenceError`).
    *   **Output:** The evaluation layer capable of converting uncharacterized atomic structures into high-fidelity training data (energies, forces, stresses).

*   **Cycle 05: The ACE Trainer & Dataset Management**
    *   **Focus:** Implement the `PacemakerWrapper` fulfilling the `AbstractTrainer` contract.
    *   **Implementation:** Build `src/trainers/ace_trainer.py` to manage the active learning dataset (converting ASE Atoms to Pacemaker formats) and orchestrate the `pace_train` subprocess. Implement the active set selection logic using D-Optimality to minimize computational overhead during subsequent training phases. Ensure the output is a valid, securely hashed `.yace` potential file.
    *   **Output:** The training layer capable of continuously refining the interatomic potential using the newly acquired Oracle data.

*   **Cycle 06: Orchestration, Validation & Tutorials**
    *   **Focus:** Implement the central `ActiveLearningOrchestrator`, the `Validator`, and build the interactive user tutorials.
    *   **Implementation:** Build `src/core/orchestrator.py` to tie all previous cycles together. Implement the strict state machine loop: setup isolated temporary directories, execute exploration, handle `DynamicsHaltInterrupt`, trigger candidate generation, compute DFT forces, train the new potential, validate it (`src/validators/validator.py`), and finally deploy it atomically using `shutil.move`. Create the `tutorials/UAT_AND_TUTORIAL.py` Marimo notebook to provide a compelling, interactive demonstration of the entire pipeline, including the FePt/MgO interface generation and the OTF halt-and-heal mechanisms in a simulated "Mock Mode".
    *   **Output:** The fully functional, automated MLIP pipeline and its accompanying user documentation.

## 6. Test Strategy

The testing strategy emphasizes isolation, security, and rigorous validation of the Orchestrator's state transitions without relying on massive, long-running external physics calculations.

*   **Testing Infrastructure:** The test suite will heavily utilize `pytest` fixtures, specifically `tmp_path`, to ensure all file I/O operations are strictly sandboxed and do not pollute the host filesystem. `unittest.mock` and `pytest.MonkeyPatch` will be used extensively to simulate the behavior of external binaries (LAMMPS, Quantum ESPRESSO, Pacemaker) without actually executing them.

*   **Cycle 01 (Config):** Unit tests will focus on Pydantic validation. We will verify that `ProjectConfig` correctly rejects invalid `.env` files (e.g., symlinks, world-writable files) and that path traversal attempts (e.g., `../etc/shadow`) in directory configurations are strictly caught and raise appropriate `ValueError` exceptions.

*   **Cycle 02 (Dynamics):** Unit tests will mock the `subprocess.run` calls to LAMMPS and EON. We will specifically test the OTF detection logic by feeding the `MDInterface` mock dump files and verifying it correctly extracts the expected high-gamma atomic environments and returns a valid `HaltInfo` object.

*   **Cycle 03 (Generators):** Unit tests will verify the mathematical correctness of the structure perturbations. We will use ASE to assert that the `generate_interface` function correctly constructs the required boundary logic (e.g., verifying the total number of atoms and the proper alignment of the Fe and Mg terminating faces) when provided an `InterfaceTarget`.

*   **Cycle 04 (Oracle):** Unit tests will mock the ASE `Espresso` calculator. We will verify that the `DFTManager` correctly applies periodic embedding to small clusters and that it appropriately handles simulated `SCF` convergence failures by raising `OracleConvergenceError`.

*   **Cycle 05 (Trainer):** Unit tests will mock the `pace_train` execution. We will verify that the `PacemakerWrapper` correctly formats the dataset, handles the D-Optimality active set selection, and enforces the strict file size limits (e.g., `< 100MB`) and structural integrity checks (SHA256 hashing) when processing the resulting `.yace` potential file.

*   **Cycle 06 (Orchestrator & UAT):** Integration and End-to-End (E2E) testing. We will test the `ActiveLearningOrchestrator` using a complete suite of mock classes for the Dynamics, Oracle, and Trainer interfaces. This allows us to rapidly simulate multiple full iterations of the active learning loop, verifying that the Orchestrator correctly handles a `DynamicsHaltInterrupt`, isolates work within temporary directories, and performs atomic deployments of the final potential. The Marimo tutorial notebook will be executed in a headless CI environment to ensure no Python exceptions occur during the user-facing demonstration.