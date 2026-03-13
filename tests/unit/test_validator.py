from pathlib import Path
from unittest.mock import MagicMock, patch

from src.domain_models.config import MaterialConfig, ValidationConfig
from src.validators.validator import Validator


def test_validator_pass() -> None:
    config = ValidationConfig()
    material = MaterialConfig()
    validator = Validator(config, material)

    with patch.object(validator, "_check_phonons", return_value=True):
        result = validator.validate(Path("dummy.yace"))

        assert result["passed"] is True
        assert result["status"] == "PASS"
        assert result["reason"] == ""
        assert result["metrics"]["rmse_energy"] == 1.5
        assert result["metrics"]["rmse_force"] == 0.04


def test_validator_fail_phonons() -> None:
    config = ValidationConfig()
    material = MaterialConfig()
    validator = Validator(config, material)

    with patch.object(validator, "_check_phonons", return_value=False):
        result = validator.validate(Path("dummy.yace"))

        assert result["passed"] is False
        assert result["status"] == "FAIL"
        assert result["reason"] == "Validation metrics failed (RMSE or Phonons)."
        assert result["metrics"]["rmse_energy"] == 1.5
        assert result["metrics"]["rmse_force"] == 0.04
        assert result["metrics"]["phonon_stable"] is False


def test_validator_fail_rmse() -> None:
    config = ValidationConfig()
    material = MaterialConfig()
    validator = Validator(config, material)

    mock_config = MagicMock()
    from unittest.mock import PropertyMock

    type(mock_config).rmse_energy_threshold = PropertyMock(side_effect=[2.0, 1.0])
    type(mock_config).rmse_force_threshold = PropertyMock(return_value=0.05)

    validator.config = mock_config

    with patch.object(validator, "_check_phonons", return_value=True):
        result = validator.validate(Path("dummy.yace"))

        assert result["passed"] is False
        assert result["status"] == "FAIL"


@patch.dict("sys.modules", {"pyacemaker.calculator": None})
def test_validator_check_phonons_no_pacemaker() -> None:
    config = ValidationConfig()
    material = MaterialConfig()
    validator = Validator(config, material)

    result = validator._check_phonons(Path("dummy.yace"))

    assert result is True  # simulated success since pyacemaker is missing
