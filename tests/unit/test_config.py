from pathlib import Path

import pytest
from pydantic import ValidationError

from src.domain_models.config import (
    DynamicsConfig,
    OracleConfig,
    ProjectConfig,
    SystemConfig,
    TrainerConfig,
    ValidatorConfig,
)


def test_system_config_valid() -> None:
    config = SystemConfig(elements=["Fe", "Pt"])
    assert config.elements == ["Fe", "Pt"]
    assert config.baseline_potential == "zbl"


def test_system_config_invalid() -> None:
    with pytest.raises(ValidationError):
        SystemConfig(elements=[])  # Empty list violates min_length=1

    with pytest.raises(ValidationError):
        SystemConfig(elements=["Fe"], extra_field="bad")  # type: ignore[call-arg]


def test_dynamics_config_valid() -> None:
    config = DynamicsConfig(uncertainty_threshold=10.0)
    assert config.uncertainty_threshold == 10.0


def test_oracle_config_invalid() -> None:
    with pytest.raises(ValidationError):
        OracleConfig(kspacing=-0.1)  # gt=0.0 constraint violated


def test_project_config(tmp_path: Path) -> None:
    # Should be valid with minimal required setup (since most have defaults)
    config = ProjectConfig(
        project_root=tmp_path,
        system=SystemConfig(elements=["Fe", "O"]),
        dynamics=DynamicsConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        validator=ValidatorConfig(),
    )
    assert config.system.elements == ["Fe", "O"]
    assert config.system.baseline_potential == "zbl"
