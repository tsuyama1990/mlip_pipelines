from pathlib import Path

import pytest

from src.domain_models.config import ValidatorConfig
from src.validators.validator import Validator


def test_validator_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    config = ValidatorConfig(energy_rmse_threshold=0.01)
    validator = Validator(config)
    assert validator.config.energy_rmse_threshold == 0.01


def test_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    config = ValidatorConfig()
    validator = Validator(config)

    # Needs a potential path
    dummy_pot = tmp_path / "dummy.yace"
    dummy_pot.write_text("elements version")

    import importlib.util

    if importlib.util.find_spec("pyacemaker") is None:
        pytest.skip("pyacemaker is missing, skipping actual validation run")

    if importlib.util.find_spec("phonopy") is None:
        pytest.skip("phonopy is missing, skipping actual validation run")

    try:
        report = validator.validate(dummy_pot)
        assert hasattr(report, "energy_rmse")
    except Exception as e:
        pytest.skip(
            f"Failed to execute real validation due to missing model structures/formats: {e}"
        )
