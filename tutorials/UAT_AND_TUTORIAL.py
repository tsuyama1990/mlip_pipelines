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

    # We use a temporary directory for the tutorial output
    tutorial_dir = Path(tempfile.mkdtemp(prefix="mlip_tutorial_"))

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

        # Mock heavy external dependencies to allow rapid execution
        mock_pyacemaker = type("pyacemaker", (), {"pyacemaker": True})
        sys.modules["pyacemaker.calculator"] = mock_pyacemaker # type: ignore

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
            mo.md("⚠️ Heavy dependencies missing. Simulating AL loop...")

            # Create dummy potentials dir and yace file
            pot_dir = config.project_root / "potentials"
            pot_dir.mkdir(exist_ok=True, parents=True)
            dummy_yace = pot_dir / "generation_001.yace"
            dummy_yace.touch()

            # Simulate state changes
            orchestrator.iteration = 1
            result_path = str(dummy_yace)

            mo.md(f"**Mock Cycle Completed!** Potential created at: `{result_path}`")
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
        # Generate a dummy HTML report to show the UI
        html_content = """
        <html>
            <head><title>MLIP Validation Report</title></head>
            <body style="font-family: sans-serif; padding: 20px;">
                <h1>Quality Assurance: Validation Report</h1>
                <h2>Test Set RMSE</h2>
                <ul>
                    <li><strong>Energy RMSE:</strong> 0.0015 eV/atom <span style="color: green;">(PASS)</span></li>
                    <li><strong>Force RMSE:</strong> 0.042 eV/A <span style="color: green;">(PASS)</span></li>
                    <li><strong>Stress RMSE:</strong> 0.08 GPa <span style="color: green;">(PASS)</span></li>
                </ul>
                <h2>Mechanical Stability</h2>
                <ul>
                    <li><strong>Born Criteria:</strong> <span style="color: green;">Stable</span></li>
                    <li><strong>Elastic Constants:</strong> C11=250 GPa, C12=150 GPa, C44=100 GPa</li>
                </ul>
                <h2>Phonon Dispersion</h2>
                <p>No imaginary frequencies detected.</p>
                <div style="width: 100%; height: 200px; background-color: #f0f0f0; border: 1px solid #ccc; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #666;">[Mock Phonon Dispersion Plot]</span>
                </div>
            </body>
        </html>
        """
        report_path.write_text(html_content, encoding="utf-8")
        mo.md("Generated mock validation report.")
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
