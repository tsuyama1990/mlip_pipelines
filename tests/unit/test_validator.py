from pathlib import Path
from unittest.mock import patch

from src.domain_models.config import MaterialConfig, ValidationConfig
from src.validators.validator import Validator


def test_validator_pass(mock_material_config: MaterialConfig) -> None:
    config = ValidationConfig(rmse_energy_offset=0.5, rmse_force_offset=0.01)
    material = mock_material_config
    validator = Validator(config, material)

    with patch.object(validator, "_check_phonons", return_value=True):
        result = validator.validate(Path("dummy.yace"))

        assert result["passed"] is True
        assert result["status"] == "PASS"
        assert result["reason"] == ""
        assert result["metrics"]["rmse_energy"] == 1.5
        assert result["metrics"]["rmse_force"] == 0.04


def test_validator_fail_phonons(mock_material_config: MaterialConfig) -> None:
    config = ValidationConfig(rmse_energy_offset=0.5, rmse_force_offset=0.01)
    material = mock_material_config
    validator = Validator(config, material)

    with patch.object(validator, "_check_phonons", return_value=False):
        result = validator.validate(Path("dummy.yace"))

        assert result["passed"] is False
        assert result["status"] == "FAIL"
        assert result["reason"] == "Validation metrics failed (RMSE or Phonons)."
        assert result["metrics"]["rmse_energy"] == 1.5
        assert result["metrics"]["rmse_force"] == 0.04
        assert result["metrics"]["phonon_stable"] is False


def test_validator_fail_rmse(mock_material_config: MaterialConfig) -> None:
    # Set the offset such that threshold - offset > threshold
    # Using negative offset achieves this to cause a fail
    config = ValidationConfig(rmse_energy_offset=-1.0, rmse_force_offset=-1.0)
    material = mock_material_config
    validator = Validator(config, material)

    with patch.object(validator, "_check_phonons", return_value=True):
        result = validator.validate(Path("dummy.yace"))

        assert result["passed"] is False
        assert result["status"] == "FAIL"


@patch.dict("sys.modules", {"pyacemaker.calculator": None})
def test_validator_check_phonons_no_pacemaker(mock_material_config: MaterialConfig) -> None:
    config = ValidationConfig()
    material = mock_material_config
    validator = Validator(config, material)

    import pytest

    with pytest.raises(RuntimeError, match="Missing required PACE bindings for validation."):
        validator._check_phonons(Path("dummy.yace"))
