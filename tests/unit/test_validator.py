import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from src.domain_models.config import ValidatorConfig
from src.validators import stability_tests
from src.validators.validator import Validator


def test_validator_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    config = ValidatorConfig(energy_rmse_threshold=0.01)
    validator = Validator(config)
    assert validator.config.energy_rmse_threshold == 0.01


class MockCalc(Calculator):  # type: ignore
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def calculate(
        self, atoms: Any = None, properties: Any = None, system_changes: Any = None
    ) -> None:
        self.results = {
            "energy": -10.0,
            "forces": np.zeros((len(atoms) if atoms else 0, 3)),
            "stress": np.zeros(6),
        }


def test_validator_rmse_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No test dataset provided -> skips RMSE
    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": MockCalc})
    )
    config = ValidatorConfig(test_dataset_path=None)
    validator = Validator(config)

    # mock stability
    monkeypatch.setattr(validator, "_check_phonopy_stability", lambda *args: True)
    monkeypatch.setattr(stability_tests, "check_mechanical_stability", lambda *args: True)

    pot = tmp_path / "pot.yace"
    pot.write_text("elements version b_functions")

    report = validator.validate(pot)
    assert report.passed is True
    assert report.energy_rmse == 0.0
    assert report.force_rmse == 0.0
    assert report.stress_rmse == 0.0


def test_validator_rmse_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "test.extxyz"
    from ase.io import write

    atoms = Atoms("Fe", positions=[(0, 0, 0)], cell=[5, 5, 5], pbc=True)
    atoms.info["energy"] = -10.0
    atoms.arrays["forces"] = np.zeros((1, 3))
    atoms.info["stress"] = np.zeros(6)
    write(str(dataset), [atoms], format="extxyz")

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": MockCalc})
    )
    config = ValidatorConfig(test_dataset_path=str(dataset))
    validator = Validator(config)

    # mock stability
    monkeypatch.setattr(validator, "_check_phonopy_stability", lambda *args: True)
    monkeypatch.setattr(stability_tests, "check_mechanical_stability", lambda *args: True)

    pot = tmp_path / "pot.yace"
    pot.write_text("elements version b_functions")

    report = validator.validate(pot)
    assert report.passed is True
    assert report.energy_rmse == 0.0


def test_validator_rmse_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "test.extxyz"
    from ase.io import write

    atoms = Atoms("Fe", positions=[(0, 0, 0)], cell=[5, 5, 5], pbc=True)
    atoms.info["energy"] = 1000.0  # large error
    atoms.arrays["forces"] = np.zeros((1, 3))
    atoms.info["stress"] = np.zeros(6)
    write(str(dataset), [atoms], format="extxyz")

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": MockCalc})
    )
    config = ValidatorConfig(test_dataset_path=str(dataset))
    validator = Validator(config)

    # mock stability
    monkeypatch.setattr(validator, "_check_phonopy_stability", lambda *args: True)
    monkeypatch.setattr(stability_tests, "check_mechanical_stability", lambda *args: True)

    pot = tmp_path / "pot.yace"
    pot.write_text("elements version b_functions")

    report = validator.validate(pot)
    assert report.passed is False
    assert "Thresholds exceeded or instability detected." in str(report.reason)


def test_validator_stability_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": MockCalc})
    )
    config = ValidatorConfig(test_dataset_path=None)
    validator = Validator(config)

    # mock stability to fail
    monkeypatch.setattr(validator, "_check_phonopy_stability", lambda *args: False)
    monkeypatch.setattr(stability_tests, "check_mechanical_stability", lambda *args: True)

    pot = tmp_path / "pot.yace"
    pot.write_text("elements version b_functions")

    report = validator.validate(pot)
    assert report.passed is False
    assert "Thresholds exceeded or instability detected." in str(report.reason)


def test_validator_exception_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": MockCalc})
    )
    config = ValidatorConfig(test_dataset_path=None)
    validator = Validator(config)

    def fail_eval(*args: Any) -> bool:
        msg = "Calculation crashed"
        raise RuntimeError(msg)

    monkeypatch.setattr(validator, "_check_phonopy_stability", fail_eval)

    pot = tmp_path / "pot.yace"
    pot.write_text("elements version b_functions")

    with pytest.raises(RuntimeError, match="Validation execution failed: Calculation crashed"):
        validator.validate(pot)


def test_validator_invalid_potential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": MockCalc})
    )
    config = ValidatorConfig()
    validator = Validator(config)

    pot = tmp_path / "bad.txt"
    pot.touch()
    with pytest.raises(ValueError, match="Potential file must have .yace extension"):
        validator.validate(pot)

    pot2 = tmp_path / "pot.yace"
    pot2.write_text("invalid")
    with pytest.raises(ValueError, match="does not appear to be a valid YACE format"):
        validator.validate(pot2)
