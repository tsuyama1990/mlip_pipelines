import re

with open('tutorials/UAT_AND_TUTORIAL.py', 'r') as f:
    content = f.read()

# 1. Fix the Mock implementation to properly mock pyacemaker
new_mock = """        import sys
        import unittest.mock
        from ase import Atoms
        from ase.calculators.calculator import Calculator

        # Proper Calculator Mock
        class MockPaceCalculator(Calculator):
            implemented_properties = ['energy', 'forces', 'free_energy']
            def calculate(self, atoms=None, properties=None, system_changes=None):
                Calculator.calculate(self, atoms, properties, system_changes)
                self.results['energy'] = -10.0 * len(atoms) if atoms else 0.0
                if atoms:
                    import numpy as np
                    self.results['forces'] = np.zeros((len(atoms), 3))

        # Setup module structure
        mock_module = type("pyacemaker", (), {"pyacemaker": True})
        mock_module.calculator = type("calculator", (), {"PyACEMakerCalculator": MockPaceCalculator})
        sys.modules["pyacemaker"] = mock_module
        sys.modules["pyacemaker.calculator"] = mock_module.calculator
"""
content = re.sub(r"        import sys\n        import unittest\.mock\n.*?\n        sys\.modules\[\"pyacemaker\.calculator\"\] = mock_pyacemaker # type: ignore", new_mock, content, flags=re.DOTALL)


# 2. Fix the AL Loop simulation
new_sim = """        if not can_run:
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
"""
content = re.sub(r"        if not can_run:\n            mo.md\(\"⚠️ Heavy dependencies missing\. Simulating AL loop\.\.\.\"\)\n.*?\n            mo.md\(f\"\*\*Mock Cycle Completed!\*\* Potential created at: `\{result_path\}`\"\)", new_sim, content, flags=re.DOTALL)


# 3. Fix Validator Mock
new_val = """    if use_mock:
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
"""
content = re.sub(r"    if use_mock:\n        # Generate a dummy HTML report to show the UI\n        html_content = \"\"\"\n        <html>.*?\n        report_path\.write_text\(html_content, encoding=\"utf-8\"\)\n        mo.md\(\"Generated mock validation report\.\"\)", new_val, content, flags=re.DOTALL)

# 4. Fix Temp Directory cleanup
content = content.replace(
    'tutorial_dir = Path(tempfile.mkdtemp(prefix="mlip_tutorial_"))',
    '''import atexit
    import shutil
    tutorial_dir = Path(tempfile.mkdtemp(prefix="mlip_tutorial_"))
    atexit.register(lambda: shutil.rmtree(tutorial_dir, ignore_errors=True))'''
)


with open('tutorials/UAT_AND_TUTORIAL.py', 'w') as f:
    f.write(content)
