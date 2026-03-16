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
    from src.domain_models.config import DistillationConfig, LoopStrategyConfig, CutoutConfig
    return ProjectConfig(
        project_root=tmp_path,
        system=mock_system_config,
        dynamics=DynamicsConfig(trusted_directories=[], project_root=str(tmp_path)),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig(),
        distillation_config=DistillationConfig(temp_dir=str(tmp_path), output_dir=str(tmp_path), model_storage_path=str(tmp_path)),
        loop_strategy=LoopStrategyConfig(replay_buffer_size=1000, checkpoint_interval=5, timeout_seconds=3600),
        cutout_config=CutoutConfig(),
    )
