# ruff: noqa: S108
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
        dynamics=DynamicsConfig(trusted_directories=[], project_root=str(tmp_path)),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig(),
        distillation_config=__import__(
            "src.domain_models.config", fromlist=["DistillationConfig"]
        ).DistillationConfig(temp_dir="/tmp", output_dir="/tmp", model_storage_path="/tmp"),
        loop_strategy=__import__(
            "src.domain_models.config", fromlist=["LoopStrategyConfig"]
        ).LoopStrategyConfig(replay_buffer_size=10, checkpoint_interval=5, timeout_seconds=3600),
    )
