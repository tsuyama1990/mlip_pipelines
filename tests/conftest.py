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


from pathlib import Path

@pytest.fixture
def mock_project_config(mock_system_config: SystemConfig, tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        project_root=tmp_path,
        system=mock_system_config,
        dynamics=DynamicsConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        validator=ValidatorConfig(),
    )
