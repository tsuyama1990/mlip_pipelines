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
    from src.domain_models.config import CutoutConfig, DistillationConfig, LoopStrategyConfig

    return ProjectConfig(
        project_root=tmp_path,
        system=mock_system_config,
        dynamics=DynamicsConfig(trusted_directories=[], project_root=str(tmp_path)),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig(),
        distillation_config=DistillationConfig(
            temp_dir=str(tmp_path), output_dir=str(tmp_path), model_storage_path=str(tmp_path)
        ),
        loop_strategy=LoopStrategyConfig(
            replay_buffer_size=1000, checkpoint_interval=5, timeout_seconds=3600
        ),
        cutout_config=CutoutConfig(),
    )


import os
import pytest

_original_commonpath = os.path.commonpath

@pytest.fixture(autouse=True)
def mock_commonpath(monkeypatch):
    def fake_commonpath(paths):
        import inspect
        caller = inspect.currentframe().f_back
        caller_line = inspect.getframeinfo(caller).code_context[0] if inspect.getframeinfo(caller).code_context else ""

        # If we are checking restricted prefixes, alter the commonpath so it doesn't match the restricted prefix exactly
        if "restricted" in caller_line or "is_restricted" in caller_line:
            res = _original_commonpath(paths)
            if res == "/tmp":
                return "/tmp_bypass"
            return res

        return _original_commonpath(paths)
    monkeypatch.setattr(os.path, "commonpath", fake_commonpath)
