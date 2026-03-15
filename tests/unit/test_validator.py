import sys
import typing
from pathlib import Path
from typing import Never

import numpy as np
import pytest

from src.domain_models.config import ValidatorConfig
from src.validators.validator import Validator


def _setup_mock_pyacemaker(monkeypatch, energy=1.0, forces=None, stress=None):

    from ase.calculators.calculator import Calculator
    class UnitMockCalc(Calculator):
        implemented_properties: list[str] = ["energy", "forces", "stress"]
        def __init__(self) -> None:
            super().__init__()
            self.energy = energy
            self.forces = forces if forces is not None else np.array([[0.05, 0.05, 0.05]])
            self.stress = stress if stress is not None else np.array([0.1]*6)
        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": self.energy,
                "forces": self.forces,
                "stress": self.stress,
            }

    class MockPyacemakerModule:
        @staticmethod
        def pyacemaker(path: str) -> typing.Any:
            return UnitMockCalc()

    sys.modules["pyacemaker"] = MockPyacemakerModule()
    sys.modules["pyacemaker.calculator"] = MockPyacemakerModule()
    monkeypatch.setattr("src.validators.validator.Validator._check_phonopy_stability", lambda self, a, c: True)
    monkeypatch.setattr("src.validators.stability_tests.check_mechanical_stability", lambda a, c: True)


def test_compute_metrics_with_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _setup_mock_pyacemaker(monkeypatch)

    config = ValidatorConfig(test_dataset_path=str(tmp_path / "test.extxyz"))

    dummy_pot = tmp_path / "dummy.yace"
    dummy_pot.write_text("elements version")

    dataset_path = tmp_path / "test.extxyz"
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    atoms = Atoms("Fe", positions=[(0, 0, 0)])
    atoms.calc = SinglePointCalculator(atoms, energy=1.0, forces=np.array([[0.05, 0.05, 0.05]]), stress=np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0]))
    write(str(dataset_path), atoms, format="extxyz")

    validator = Validator(config)
    e, f, s, ps, ms = validator._compute_metrics(dummy_pot)
    assert isinstance(ps, bool)


def test_validate_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _setup_mock_pyacemaker(monkeypatch, energy=1.0)

    config = ValidatorConfig(
        energy_rmse_threshold=0.1, force_rmse_threshold=0.2, stress_rmse_threshold=0.3, test_dataset_path=str(tmp_path / "test.extxyz")
    )

    dataset_path = tmp_path / "test.extxyz"
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    atoms = Atoms("Fe", positions=[(0, 0, 0)])
    atoms.calc = SinglePointCalculator(atoms, energy=1.0, forces=np.array([[0.05, 0.05, 0.05]]), stress=np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0]))
    write(str(dataset_path), atoms, format="extxyz")

    validator = Validator(config)
    dummy_pot = tmp_path / "dummy.yace"
    dummy_pot.write_text("elements version")

    report = validator.validate(dummy_pot)
    assert report.passed is True


def test_validate_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _setup_mock_pyacemaker(monkeypatch, energy=100.0)

    config = ValidatorConfig(energy_rmse_threshold=0.01, test_dataset_path=str(tmp_path / "test.extxyz"))

    dataset_path = tmp_path / "test.extxyz"
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    atoms = Atoms("Fe", positions=[(0, 0, 0)])
    atoms.calc = SinglePointCalculator(atoms, energy=1.0, forces=np.array([[0.05, 0.05, 0.05]]), stress=np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0]))
    write(str(dataset_path), atoms, format="extxyz")

    validator = Validator(config)
    dummy_pot = tmp_path / "dummy.yace"
    dummy_pot.write_text("elements version")

    report = validator.validate(dummy_pot)
    assert report.passed is False

def test_validate_runtime_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MockPyacemakerModuleFailure:
        @staticmethod
        def pyacemaker(path) -> Never:
            msg = "Mock error"
            raise RuntimeError(msg)

    sys.modules["pyacemaker"] = MockPyacemakerModuleFailure()
    sys.modules["pyacemaker.calculator"] = MockPyacemakerModuleFailure()

    config = ValidatorConfig()
    validator = Validator(config)

    dummy_pot = tmp_path / "dummy.yace"
    dummy_pot.write_text("elements version")

    with pytest.raises(RuntimeError, match="Validation execution failed: Mock error"):
        validator.validate(dummy_pot)

def test_validator_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_mock_pyacemaker(monkeypatch)
    config = ValidatorConfig(energy_rmse_threshold=0.01)
    validator = Validator(config)
    assert validator.config.energy_rmse_threshold == 0.01

def test_check_file_format_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_mock_pyacemaker(monkeypatch)
    config = ValidatorConfig()
    validator = Validator(config)

    missing_path = tmp_path / "missing.yace"
    with pytest.raises(FileNotFoundError, match="Potential file not found or is not a file"):
        validator._check_file_format(missing_path)

def test_check_file_format_bad_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_mock_pyacemaker(monkeypatch)
    config = ValidatorConfig()
    validator = Validator(config)

    bad_ext_path = tmp_path / "dummy.txt"
    bad_ext_path.write_text("elements version")
    with pytest.raises(ValueError, match="Potential file must have .yace extension"):
        validator._check_file_format(bad_ext_path)

def test_check_file_format_invalid_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_mock_pyacemaker(monkeypatch)
    config = ValidatorConfig()
    validator = Validator(config)

    bad_content_path = tmp_path / "dummy.yace"
    bad_content_path.write_text("invalid content here")
    with pytest.raises(ValueError, match="does not appear to be a valid YACE format"):
        validator._check_file_format(bad_content_path)
