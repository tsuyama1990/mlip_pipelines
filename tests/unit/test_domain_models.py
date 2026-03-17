import pytest
from pydantic import ValidationError

from src.domain_models.config import (
    ProjectConfig,
)
from src.domain_models.dtos import MaterialFeatures
from src.domain_models.gui_schemas import WorkflowIntentConfig


def test_material_features_valid() -> None:
    feat = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    assert feat.elements == ["Fe", "Pt"]


def _create_base_project_config(tmp_path) -> dict:
    (tmp_path / "pyproject.toml").touch()

    return {
        "project_root": str(tmp_path),
        "system": {"elements": ["Fe"]},
        "dynamics": {"trusted_directories": [str(tmp_path)], "project_root": str(tmp_path)},
        "oracle": {},
        "trainer": {"trusted_directories": [str(tmp_path)]},
        "validator": {},
        "distillation_config": {
            "temp_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "model_storage_path": str(tmp_path),
        },
        "loop_strategy": {
            "replay_buffer_size": 500,
            "checkpoint_interval": 10,
            "timeout_seconds": 3600
        }
    }


def test_workflow_intent_translation_speed(tmp_path) -> None:
    data = _create_base_project_config(tmp_path)
    data["intent"] = {
        "target_material": "Fe",
        "accuracy_speed_tradeoff": 1
    }

    config = ProjectConfig(**data)

    # At tradeoff=1, threshold = 0.16444 - 0.01444 = 0.150
    assert config.distillation_config.uncertainty_threshold == 0.150
    # Buffer size = 100 + (1-1)*544 = 100
    assert config.loop_strategy.replay_buffer_size == 100
    # Max retries = 3 + (1 // 2) = 3
    assert config.loop_strategy.max_retries == 3


def test_workflow_intent_translation_accuracy(tmp_path) -> None:
    data = _create_base_project_config(tmp_path)
    data["intent"] = {
        "target_material": "Fe",
        "accuracy_speed_tradeoff": 10
    }

    config = ProjectConfig(**data)

    # At tradeoff=10, threshold = 0.16444 - 0.1444 = 0.02
    assert config.distillation_config.uncertainty_threshold == 0.020
    # Buffer size = 100 + (10-1)*544 = 4996
    assert config.loop_strategy.replay_buffer_size == 4996
    # Max retries = 3 + (10 // 2) = 8
    assert config.loop_strategy.max_retries == 8


def test_workflow_intent_security_validation() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WorkflowIntentConfig(target_material="../../etc/passwd", accuracy_speed_tradeoff=5)

    assert "Path traversal characters are not allowed" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        WorkflowIntentConfig(target_material="Fe; rm -rf /", accuracy_speed_tradeoff=5)

    assert "Shell injection characters are not allowed" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        WorkflowIntentConfig(target_material="Fe", accuracy_speed_tradeoff=15)

    assert "Input should be less than or equal to 10" in str(exc_info.value)
