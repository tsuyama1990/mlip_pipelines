import marimo

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    return mo,

@app.cell
def __(mo):
    mo.md(
        """
        # MLIP Pipeline Interactive Tutorial & UAT

        Welcome to the Active Learning Orchestrator interactive tutorial.
        """
    )

@app.cell
def __():
    USE_MOCK = True  # Toggle for actual HPC execution vs CI/Test dummy
    return USE_MOCK,

@app.cell
def __(USE_MOCK, mo):
    mo.md("## Configuration")

@app.cell
def __(USE_MOCK):
    import os
    import sys
    from pathlib import Path

    # Path patching for headless test execution directly
    sys.path.insert(0, str(Path.cwd()))

    from src.domain_models.config import (
        DynamicsConfig,
        OracleConfig,
        ProjectConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )

    # Base configuration mimicking FePt / MgO system
    config = ProjectConfig(
        system=SystemConfig(elements=["Fe", "Pt", "Mg", "O"], baseline_potential="zbl"),
        dynamics=DynamicsConfig(uncertainty_threshold=5.0, md_steps=1000),
        oracle=OracleConfig(kspacing=0.1, smearing_width=0.02, pseudo_dir=str(Path.cwd())),
        trainer=TrainerConfig(max_epochs=2, active_set_size=10),
        validator=ValidatorConfig(energy_rmse_threshold=0.05),
        project_root=Path.cwd()
    )
    return config,

@app.cell
def __(config, USE_MOCK, mo):
    from src.core.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)

    if USE_MOCK:
        # Patching to allow local headless run
        class MockMD:
            def __init__(self, *args, **kwargs): pass
            def run_exploration(self, *args, **kwargs):
                return {"halted": True, "dump_file": config.project_root / "dummy_dump"}
            def extract_high_gamma_structures(self, *args, **kwargs):
                from ase import Atoms
                return [Atoms("Fe", positions=[(0, 0, 0)])]

        class MockOracle:
            def __init__(self, *args, **kwargs): pass
            def compute_batch(self, batch, *args, **kwargs):
                return batch

        class MockTrainer:
            def __init__(self, *args, **kwargs): pass
            def select_local_active_set(self, candidates, anchor, n):
                return candidates[:n]
            def update_dataset(self, new_data, dataset_path):
                return dataset_path
            def train(self, dataset, initial_potential, output_dir):
                pot = output_dir / "new_pot.yace"
                pot.parent.mkdir(parents=True, exist_ok=True)
                pot.write_text("dummy potential")
                return pot

        class MockValidator:
            def __init__(self, *args, **kwargs): pass
            def validate(self, *args, **kwargs):
                from src.domain_models.dtos import ValidationReport
                return ValidationReport(passed=True, energy_rmse=0.001, force_rmse=0.01, stress_rmse=0.05, phonon_stable=True, mechanically_stable=True)

        orchestrator.md_engine = MockMD()
        orchestrator.oracle = MockOracle()
        orchestrator.trainer = MockTrainer()
        orchestrator.validator = MockValidator()

        # Patch generator just to make it fast
        class MockGenerator:
            def __init__(self, *args, **kwargs): pass
            def generate_local_candidates(self, s0, n=20):
                from ase import Atoms
                return [Atoms("Fe", positions=[(0, 0, 0)])]*n
        orchestrator.structure_generator = MockGenerator()

    mo.md("Orchestrator Initialized")
    return orchestrator,

@app.cell
def __(orchestrator, mo):
    mo.md("## Run Phase 1: Zero-Config Active Learning Cycle")

    # Run the cycle
    result_pot = orchestrator.run_cycle()

    output = {
        "status": "Success",
        "final_potential": str(result_pot),
        "iteration": orchestrator.iteration,
        "FePt/MgO Interface Energy": 1.25, # Dummy values for the Aha moment calculation
        "FePt Order Parameter": 0.85
    }

    return result_pot, output

@app.cell
def __(output, mo):
    mo.ui.table([output])

if __name__ == "__main__":
    app.run()
