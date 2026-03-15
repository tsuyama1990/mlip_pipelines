import sys
from pathlib import Path
import pytest
from src.domain_models.config import ValidatorConfig
from src.domain_models.dtos import ValidationReport
from src.validators.validator import Validator
from src.validators.reporter import Reporter
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


class MockCalc(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    def __init__(self, energy=1.0, forces=None, stress=None):
        super().__init__()
        self.preset_energy = energy
        self.preset_forces = forces if forces is not None else np.array([[0.1, 0.1, 0.1]])
        self.preset_stress = stress if stress is not None else np.array([0.01]*6)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": self.preset_energy,
            "forces": self.preset_forces,
            "stress": self.preset_stress,
        }

def test_uat_06_01_quality_assurance_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True}))
    # Mock pyacemaker so it returns a MockCalc
    monkeypatch.setattr(sys.modules["pyacemaker.calculator"], "pyacemaker", lambda path: MockCalc(energy=1.001, forces=np.array([[0.1, 0.1, 0.1]])))

    config = ValidatorConfig(energy_rmse_threshold=0.002, force_rmse_threshold=0.05, test_dataset_path=str(tmp_path / "test.extxyz"))

    # Create the test dataset with known ground truth
    dataset_path = tmp_path / "test.extxyz"
    dataset_path.write_text("test")

    def mock_read(path: str, index: str):
        atoms = Atoms("Fe", positions=[(0, 0, 0)])
        atoms.calc = MockCalc(energy=1.0, forces=np.array([[0.1, 0.1, 0.1]]))
        return [atoms]

    monkeypatch.setattr("ase.io.read", mock_read)
    monkeypatch.setattr("src.validators.validator.Validator._check_phonopy_stability", lambda self, a, c: True)
    monkeypatch.setattr("src.validators.stability_tests.check_mechanical_stability", lambda a, c: True)

    validator = Validator(config)

    dummy_pot = tmp_path / "potential.yace"
    dummy_pot.write_text("elements version")

    report = validator.validate(dummy_pot)

    assert report.energy_rmse <= config.energy_rmse_threshold
    assert report.force_rmse <= config.force_rmse_threshold
    assert report.passed is True

def test_uat_06_02_quality_assurance_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True}))

    # Return high error values
    monkeypatch.setattr(sys.modules["pyacemaker.calculator"], "pyacemaker", lambda path: MockCalc(energy=1.0, forces=np.array([[0.3, 0.3, 0.3]])))

    config = ValidatorConfig(energy_rmse_threshold=0.002, force_rmse_threshold=0.05, test_dataset_path=str(tmp_path / "test.extxyz"))

    # Create the test dataset with known ground truth
    dataset_path = tmp_path / "test.extxyz"
    dataset_path.write_text("test")

    def mock_read(path: str, index: str):
        atoms = Atoms("Fe", positions=[(0, 0, 0)])
        atoms.calc = MockCalc(energy=1.0, forces=np.array([[0.1, 0.1, 0.1]]))
        return [atoms]

    monkeypatch.setattr("ase.io.read", mock_read)
    monkeypatch.setattr("src.validators.validator.Validator._check_phonopy_stability", lambda self, a, c: True)
    monkeypatch.setattr("src.validators.stability_tests.check_mechanical_stability", lambda a, c: True)

    validator = Validator(config)

    dummy_pot = tmp_path / "poor_potential.yace"
    dummy_pot.write_text("elements version")

    report = validator.validate(dummy_pot)

    assert report.force_rmse > config.force_rmse_threshold
    assert report.passed is False
    assert "Thresholds exceeded or instability detected" in report.reason

def test_uat_06_03_automated_report_generation(tmp_path: Path):
    reporter = Reporter()

    report = ValidationReport(
        passed=True,
        reason=None,
        energy_rmse=0.001,
        force_rmse=0.04,
        stress_rmse=0.0,
        phonon_stable=True,
        mechanically_stable=True
    )

    save_path = tmp_path / "validation_report.html"
    reporter.generate_html_report(report, save_path)

    assert save_path.exists()
    content = save_path.read_text()
    assert "0.001" in content
    assert "0.04" in content
    assert "PASS" in content
