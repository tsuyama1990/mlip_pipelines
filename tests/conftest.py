import pytest

from src.domain_models.config import (
    DynamicsConfig,
    OracleConfig,
    ProjectConfig,
    SystemConfig,
    TrainerConfig,
    ValidatorConfig,
)


@pytest.fixture
def mock_system_config() -> SystemConfig:
    return SystemConfig(
        elements=["Fe", "Pt"],
        baseline_potential="zbl",
    )


@pytest.fixture
def mock_project_config(mock_system_config: SystemConfig) -> ProjectConfig:
    return ProjectConfig(
        system=mock_system_config,
        dynamics=DynamicsConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        validator=ValidatorConfig(),
    )
