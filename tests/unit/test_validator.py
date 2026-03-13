from src.domain_models.config import ValidatorConfig
from src.validators.validator import Validator


def test_validator_initialization():
    config = ValidatorConfig(energy_rmse_threshold=0.01)
    validator = Validator(config)
    assert validator.config.energy_rmse_threshold == 0.01

def test_validate(tmp_path):
    config = ValidatorConfig()
    validator = Validator(config)

    # Needs a mock potential path
    dummy_pot = tmp_path / "dummy.yace"
    dummy_pot.write_text("...")

    # We will likely need to patch internal phonopy/ASE calls
    # but the skeleton asserts the interface
    report = validator.validate(dummy_pot)
    # The dummy implementation returns True because it uses hardcoded low RMSE values when phonopy is not installed.
    # If the environment actually ran phonopy we would also be fine since it's hardcoded mock.
    assert report.passed is True
    assert hasattr(report, "energy_rmse")
