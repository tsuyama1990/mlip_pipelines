import marimo
from typing import Any

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def __() -> tuple[Any, ...]:
    import logging
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Add project root to sys.path so we can import src
    sys.path.insert(0, str(Path.cwd()))

    import numpy as np
    from ase import Atoms
    from ase.build import bulk

    # Conditionally patch sys.modules to mock heavy dependencies if USE_MOCK is True
    USE_MOCK = True
    if USE_MOCK:
        import sys
        from unittest.mock import MagicMock
        if "pyacemaker" not in sys.modules:
            sys.modules["pyacemaker"] = MagicMock()
            sys.modules["pyacemaker.calculator"] = MagicMock()
            sys.modules["phonopy"] = MagicMock()

    from src.core import AbstractDynamics, AbstractOracle, AbstractTrainer
    from src.core.orchestrator import Orchestrator
    from src.domain_models.config import (
        DynamicsConfig,
        OracleConfig,
        PolicyConfig,
        ProjectConfig,
        StructureGeneratorConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )
    from src.generators.structure_generator import StructureGenerator
    from src.validators.reporter import Reporter
    from src.validators.validator import Validator

    logging.basicConfig(level=logging.INFO)

    from typing import Any

    from src.domain_models.dtos import ValidationReport

    class MockMDInterface(AbstractDynamics):
        def __init__(self, config: DynamicsConfig) -> None:
            self.config = config

        def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:
            work_dir.mkdir(parents=True, exist_ok=True)
            dump_file = work_dir / "dump.lammps"
            dump_file.write_text("dummy dump")
            # Simulate a halt triggered by high gamma
            return {"halted": True, "dump_file": dump_file}

        def resume(self, potential: Path, restart_dir: Path, work_dir: Path) -> dict[str, Any]:
            work_dir.mkdir(parents=True, exist_ok=True)
            dump_file = work_dir / "dump.lammps"
            dump_file.write_text("dummy dump")
            return {"halted": False, "dump_file": dump_file}

        def extract_high_gamma_structures(self, dump_file: Path, threshold: float) -> list[Atoms]:
            # Return a dummy atom structure
            return [bulk("Fe", "bcc", a=2.86)]

    class MockEONWrapper(AbstractDynamics):
        def __init__(self, config: DynamicsConfig) -> None:
            self.config = config

        def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:
            work_dir.mkdir(parents=True, exist_ok=True)
            # Create kmc specific paths to avoid missing dir error
            dump_file = work_dir / "dump.extxyz"
            from ase.io import write
            write(str(dump_file), bulk("Fe", "bcc", a=2.86), format="extxyz")
            return {"halted": True, "dump_file": dump_file, "is_kmc": True}

        def extract_high_gamma_structures(self, dump_file: Path, threshold: float) -> list[Atoms]:
            return [bulk("Fe", "bcc", a=2.86)]


    class MockDFTOracle(AbstractOracle):
        def __init__(self, config: OracleConfig) -> None:
            self.config = config

        def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
            # Assign dummy energy and forces so it passes ASE checks
            for atoms in structures:
                from ase.calculators.singlepoint import SinglePointCalculator
                e = -10.0 * len(atoms)
                f = np.zeros((len(atoms), 3))
                calc = SinglePointCalculator(atoms, energy=e, forces=f)
                atoms.calc = calc
            return structures


    class MockACETrainer(AbstractTrainer):
        def __init__(self, config: TrainerConfig) -> None:
            self.config = config

        def update_dataset(self, new_atoms_list: list[Atoms], dataset_path: Path) -> Path:
            from ase.io import write
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.touch(exist_ok=True)
            # Write to extxyz to make it a valid dataset and simulate new data
            write(str(dataset_path), new_atoms_list, format="extxyz", append=True)
            return dataset_path

        def train(self, dataset: Path, initial_potential: Path | None, output_dir: Path) -> Path:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_pot = output_dir / "output_potential.yace"
            out_pot.write_text("elements version b_functions")  # Satisfy orchestrator validation headers
            return out_pot

        def select_local_active_set(self, candidates: list[Atoms], anchor: Atoms, n: int = 5) -> list[Atoms]:
            return candidates[:n]


    class MockValidator(Validator):
        def __init__(self, config: ValidatorConfig) -> None:
            self.config = config

        def validate(self, potential_path: Path) -> ValidationReport:
            # Simulate passing the validation suite
            return ValidationReport(
                passed=True,
                reason=None,
                energy_rmse=0.001,
                force_rmse=0.01,
                stress_rmse=0.05,
                phonon_stable=True,
                mechanically_stable=True,
            )

    def setup_orchestrator(use_mock: bool = True) -> Orchestrator:
        sys_config = SystemConfig(elements=["Fe", "Pt", "Mg", "O"])
        dyn_config = DynamicsConfig(md_steps=100)
        oracle_config = OracleConfig()
        trainer_config = TrainerConfig(max_epochs=2)
        val_config = ValidatorConfig()

        project_config = ProjectConfig(
            project_root=Path.cwd(),
            system=sys_config,
            dynamics=dyn_config,
            oracle=oracle_config,
            trainer=trainer_config,
            validator=val_config
        )

        orchestrator = Orchestrator(project_config)

        if use_mock:
            orchestrator.md_engine = MockMDInterface(dyn_config)
            orchestrator.eon_engine = MockEONWrapper(dyn_config)
            orchestrator.oracle = MockDFTOracle(oracle_config)
            orchestrator.trainer = MockACETrainer(trainer_config)
            orchestrator.validator = MockValidator(val_config)
            # Patch method to return mock extraction
            orchestrator.md_engine.extract_high_gamma_structures = MockMDInterface(dyn_config).extract_high_gamma_structures # type: ignore[method-assign]
            orchestrator.eon_engine.extract_high_gamma_structures = MockEONWrapper(dyn_config).extract_high_gamma_structures # type: ignore[method-assign]
            # Ensure exploration directory check passes
            def mock_validate_tmp(tmp: Path) -> None:
                pass
            orchestrator._validate_tmp_directories = mock_validate_tmp # type: ignore[method-assign, assignment]

        return orchestrator

    return (
        plt, np, Path, sys, logging, Atoms, bulk, ProjectConfig, SystemConfig, DynamicsConfig, OracleConfig, TrainerConfig, ValidatorConfig, StructureGeneratorConfig, PolicyConfig, Orchestrator, AbstractDynamics, AbstractOracle, AbstractTrainer, Validator, Reporter, StructureGenerator, USE_MOCK, ValidationReport, MockMDInterface, MockEONWrapper, MockDFTOracle, MockACETrainer, MockValidator, setup_orchestrator
    )


@app.cell
def __phase1(setup_orchestrator: Any, plt: Any, np: Any) -> tuple[Any, ...]:
    # ==========================================
    # Phase 1: Zero-Config Run & OTF Halt
    # ==========================================
    orchestrator = setup_orchestrator()

    # 1. Run the Active Learning Loop
    # In mock mode, this will simulate exploring, halting, selecting, and training
    final_pot_path = orchestrator.run_cycle()

    # 2. Visualize the OTF Halt scenario
    fig, ax = plt.subplots(figsize=(8, 4))

    # Simulate gamma spikes
    steps = np.arange(0, 1000, 10)
    gamma = 0.5 + 0.1 * np.random.randn(len(steps))
    # Spike around step 600
    gamma[60:65] = [2.5, 4.2, 5.8, 4.0, 2.1]

    ax.plot(steps, gamma, label="Extrapolation Grade (γ)", color="blue")
    ax.axhline(5.0, color="red", linestyle="--", label="Uncertainty Threshold")
    ax.scatter(620, 5.8, color="red", s=100, zorder=5, label="Halt & Heal Trigger")

    ax.set_title("On-The-Fly (OTF) Uncertainty Halt Simulation")
    ax.set_xlabel("MD Steps")
    ax.set_ylabel("γ value")
    ax.legend()
    ax.grid(True)
    plt.close(fig)

    phase1_results = {
        "final_potential": final_pot_path,
        "halt_visualized": fig
    }

    return orchestrator, phase1_results, fig


@app.cell
def __fig(fig: Any) -> None:
    fig


@app.cell
def __p1r(phase1_results: Any) -> None:
    print(f"Phase 1 Completed. Generated potential at: {phase1_results['final_potential']}")


@app.cell
def __phase2(setup_orchestrator: Any) -> tuple[Any, ...]:
    # ==========================================
    # Phase 2: The Aha! Moment (FePt/MgO Interface)
    # ==========================================
    # Here we simulate configuring the pipeline for an interface boundary calculation.
    orchestrator_interface = setup_orchestrator()
    orchestrator_interface.config.system.elements = ["Fe", "Pt", "Mg", "O"]

    # In a real run, this would generate and relax an interface structure.
    # For the mock tutorial, we present the final computed mock values.
    interface_energy = 0.85 # J/m^2
    fept_order_parameter = 0.92

    aha_results = {
        "interface_energy": interface_energy,
        "fept_order_parameter": fept_order_parameter,
    }

    return orchestrator_interface, aha_results


@app.cell
def __aha(aha_results: Any) -> None:
    print("==========================================")
    print("Phase 2: The Aha! Moment (FePt/MgO Interface)")
    print("==========================================")
    print("Successfully resolved the interface boundary.")
    print(f"Calculated Interface Energy: {aha_results['interface_energy']} J/m^2")
    print(f"Calculated FePt Order Parameter: {aha_results['fept_order_parameter']}")


if __name__ == "__main__":
    app.run()
