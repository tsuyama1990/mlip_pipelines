from pathlib import Path

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
def mock_project_config(mock_system_config: SystemConfig, tmp_path: Path) -> ProjectConfig:
    (tmp_path / "README.md").touch()
    return ProjectConfig(
        project_root=tmp_path,
        system=mock_system_config,
        dynamics=DynamicsConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        validator=ValidatorConfig(),
    )
