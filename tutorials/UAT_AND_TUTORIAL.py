import marimo

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def _() -> tuple: # type: ignore
    import os
    from pathlib import Path

    import marimo as mo
    return mo, os, Path

@app.cell
def _(mo) -> None: # type: ignore
    mo.md(
        """
        # User Acceptance Testing and Tutorial Master Plan

        Welcome to the MLIP Pipelines tutorial! This interactive notebook demonstrates the "Zero-Config"
        simplicity and power of the MLIP creation pipeline.

        The entire Active Learning cycle (Exploration $\\rightarrow$ Selection $\\rightarrow$ Calculation $\\rightarrow$ Training $\\rightarrow$ Validation)
        will unfold dynamically.
        """
    )

@app.cell
def _(mo, os, Path) -> tuple: # type: ignore
    mo.md("## Setup & Configuration")
    use_mock = os.getenv("MLIP_USE_MOCK", "True").lower() in ("true", "1", "yes")
    mo.md(f"**Mock Mode**: `{'Enabled' if use_mock else 'Disabled'}`")

    # We use a temporary directory for the tutorial output
    import atexit
    import shutil
    import tempfile

    from src.core.orchestrator import Orchestrator
    from src.domain_models.config import (
        DynamicsConfig,
        InterfaceTarget,
        OracleConfig,
        ProjectConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )
    tutorial_dir = Path(tempfile.mkdtemp(prefix="mlip_tutorial_"))
    atexit.register(lambda: shutil.rmtree(tutorial_dir, ignore_errors=True))

    # "Zero-Config" initialization: Just specify elements and baseline
    system_cfg = SystemConfig(elements=["Fe", "Pt"], baseline_potential="zbl")

    # Set up the rest of the configuration.
    # In mock mode, we reduce iterations, steps, and grid sizes to run in seconds.
    dynamics_cfg = DynamicsConfig(
        project_root=str(tutorial_dir),
        trusted_directories=[str(tutorial_dir)],
        md_steps=10 if use_mock else 10000,
        uncertainty_threshold=2.0
    )
    oracle_cfg = OracleConfig(
        pseudo_dir=str(tutorial_dir),
        kspacing=0.2 if use_mock else 0.05,
        max_kpoints=10 if use_mock else 1000
    )
    trainer_cfg = TrainerConfig(
        max_epochs=2 if use_mock else 50,
        active_set_size=5 if use_mock else 500,
        trusted_directories=[str(tutorial_dir)]
    )
    validator_cfg = ValidatorConfig(
        validation_element="Fe",
        energy_rmse_threshold=0.1 if use_mock else 0.002
    )

    config = ProjectConfig(
        project_root=tutorial_dir,
        system=system_cfg,
        dynamics=dynamics_cfg,
        oracle=oracle_cfg,
        trainer=trainer_cfg,
        validator=validator_cfg,
    )

    mo.md(f"Project initialized in temporary directory: `{tutorial_dir}`")
    return (
        Orchestrator, DynamicsConfig, InterfaceTarget, OracleConfig, ProjectConfig,
        SystemConfig, TrainerConfig, ValidatorConfig, config, tutorial_dir, use_mock
    )

@app.cell
def _(Orchestrator, config, use_mock, mo) -> tuple: # type: ignore
    mo.md("## Scenario 1: Quick Start (The AL Loop)")

    orchestrator = Orchestrator(config)

    if use_mock:
        import sys
        import unittest.mock
        from typing import Any, ClassVar

        from ase import Atoms
        from ase.calculators.calculator import Calculator

        # Proper Calculator Mock
        class MockPaceCalculator:
            name = 'pace'
            implemented_properties: ClassVar[list[str]] = ['energy', 'forces', 'free_energy']
            def __init__(self, **kwargs: Any) -> None:
                self.results: dict[str, Any] = {}
            def calculate(self, atoms: Any = None, properties: Any = None, system_changes: Any = None) -> None:
                self.results = {}
                self.results['energy'] = -10.0 * len(atoms) if atoms else 0.0
                if atoms:
                    import numpy as np
                    self.results['forces'] = np.zeros((len(atoms), 3))

        # Setup module structure
        mock_module = type("pyacemaker", (), {"pyacemaker": True})
        mock_calc = type("calculator", (), {"PyACEMakerCalculator": MockPaceCalculator})
        setattr(mock_module, "calculator", mock_calc)
        sys.modules["pyacemaker"] = mock_module  # type: ignore
        sys.modules["pyacemaker.calculator"] = mock_calc  # type: ignore


        # Patch the orchestrator's run_cycle method if we don't have all binaries
        import importlib.util
        import shutil

        # Check if we can actually run the cycle or if we should fully mock it
        can_run = (
            shutil.which("lmp") and
            shutil.which("pace_train") and
            shutil.which("pace_activeset") and
            importlib.util.find_spec("pyacemaker") and
            importlib.util.find_spec("phonopy")
        )

        if not can_run:
            mo.md("⚠️ Heavy dependencies missing. Simulating realistic AL loop phases...")
            import time

            from ase.build import bulk

            # Phase 1: Exploration
            mo.md("* Phase 1: **Exploration** (Simulated MD run)... Found uncertain structure (γ > threshold)*")
            time.sleep(0.5)

            # Phase 2: Selection
            mo.md("* Phase 2: **Selection**... Extracted local cluster around defect.*")
            time.sleep(0.5)

            # Phase 3: Calculation (Oracle)
            mo.md("* Phase 3: **Calculation**... Mock DFT converged.*")
            time.sleep(0.5)

            # Phase 4: Training
            mo.md("* Phase 4: **Training**... Optimized ACE parameters.*")
            time.sleep(0.5)

            # Create potentials dir and dummy yace file
            pot_dir = config.project_root / "potentials"
            pot_dir.mkdir(exist_ok=True, parents=True)
            dummy_yace = pot_dir / f"generation_{orchestrator.iteration + 1:03d}.yace"
            dummy_yace.touch()

            # Simulate state changes
            orchestrator.iteration += 1
            result_path = str(dummy_yace)

            mo.md(f"**Mock AL Cycle Completed!** Iteration: {orchestrator.iteration}. Potential created at: `{result_path}`")

        else:
            mo.md("Running actual cycle (with reduced parameters)...")
            try:
                result_path = orchestrator.run_cycle()
                mo.md(f"**Cycle Completed!** Potential created at: `{result_path}`")
            except Exception as e:
                mo.md(f"**Cycle Failed:** {e}")

    else:
        mo.md("Running full Real Mode AL cycle. This may take a while...")
        try:
            result_path = orchestrator.run_cycle()
            mo.md(f"**Cycle Completed!** Potential created at: `{result_path}`")
        except Exception as e:
            mo.md(f"**Cycle Failed:** {e}")

    return orchestrator,

@app.cell
def _(config, mo, InterfaceTarget) -> tuple: # type: ignore
    mo.md("## Scenario 2: Advanced - Interface Generation")

    from src.domain_models.config import StructureGeneratorConfig
    from src.generators.structure_generator import StructureGenerator

    # We want to create an interface of FePt and MgO
    config.system.interface_target = InterfaceTarget(element1="FePt", element2="MgO")
    mo.md(f"Targeting interface: `{config.system.interface_target.element1}` / `{config.system.interface_target.element2}`")

    sg_config = StructureGeneratorConfig()
    generator = StructureGenerator(sg_config)

    interface_atoms = generator.generate_interface(config.system.interface_target)

    # Use matplotlib to plot the structure
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_atoms(interface_atoms, ax, radii=0.8, rotation=('10x,45y,0z'))
    ax.set_axis_off()
    ax.set_title(f"Generated Interface: {config.system.interface_target.element1} on {config.system.interface_target.element2}")

    mo.mpl.interactive(fig)
    return generator, interface_atoms, fig, ax

@app.cell
def _(config, use_mock, mo, tutorial_dir) -> tuple: # type: ignore
    mo.md("## Scenario 3: Validation and Reporting")

    from src.validators.reporter import Reporter
    from src.validators.validator import Validator

    # Create the validator and reporter
    validator = Validator(config.validator)
    reporter = Reporter()

    report_path = tutorial_dir / "validation_report.html"

    if use_mock:
        # Generate a realistic-looking report dynamically
        import time

        from src.domain_models.dtos import ValidationReport

        # Simulate validation processing
        time.sleep(1.0)

        # Create a mock validation report based on the configured thresholds to demonstrate success
        mock_report = ValidationReport(
            passed=True,
            energy_rmse=validator.config.energy_rmse_threshold * 0.8,
            force_rmse=validator.config.force_rmse_threshold * 0.7,
            stress_rmse=validator.config.stress_rmse_threshold * 0.9,
            phonon_stable=True,
            mechanically_stable=True
        )
        reporter.generate_html_report(mock_report, report_path)
        mo.md("Generated dynamic mock validation report based on configuration thresholds.")

    else:
        # In real mode, we would call validator.validate_potential(result_path)
        # and reporter.generate_report(...)
        mo.md("Skipping full validation in tutorial mode due to missing test dataset.")

    # Render the report in an iframe if it exists
    if report_path.exists():
        html = report_path.read_text(encoding="utf-8")
        # Replace quotes with HTML entities safely to avoid Python f-string issues
        html_safe = html.replace('"', '&quot;')
        mo.md(f'<iframe srcdoc="{html_safe}" width="100%" height="500px" style="border: 1px solid #ddd; border-radius: 4px;"></iframe>')

    return validator, reporter, report_path

if __name__ == "__main__":
    app.run()
