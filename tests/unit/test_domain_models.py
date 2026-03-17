import pytest
from pydantic import ValidationError

from src.domain_models.config import ProjectConfig
from src.domain_models.dtos import MaterialFeatures, WorkflowIntentConfig


def test_material_features_valid() -> None:
    feat = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    assert feat.elements == ["Fe", "Pt"]





def test_workflow_intent_config_validation() -> None:
    # Test valid
    intent = WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=5)
    assert intent.target_material == "Pt-Ni"
    assert intent.accuracy_speed_tradeoff == 5
    assert not intent.enable_auto_hpo

    # Test path traversal injection
    with pytest.raises(ValidationError) as exc:
        WorkflowIntentConfig(target_material="../../etc/passwd", accuracy_speed_tradeoff=5)
    assert "Path traversal characters" in str(exc.value)

    # Test accuracy_speed_tradeoff out of bounds
    with pytest.raises(ValidationError):
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=11)
    with pytest.raises(ValidationError):
        WorkflowIntentConfig(target_material="Pt-Ni", accuracy_speed_tradeoff=0)


def test_project_config_intent_translation(tmp_path) -> None:
    # Need to satisfy project_root constraints
    (tmp_path / "pyproject.toml").touch()

    # A base dictionary for ProjectConfig
    config_dict = {
        "project_root": str(tmp_path),
        "system": {"elements": ["Fe", "Pt"]},
        "dynamics": {"trusted_directories": [str(tmp_path)], "project_root": str(tmp_path)},
        "oracle": {},
        "trainer": {"trusted_directories": [str(tmp_path)]},
        "validator": {},
        "distillation_config": {
            "temp_dir": str(tmp_path / "temp"),
            "output_dir": str(tmp_path / "out"),
            "model_storage_path": str(tmp_path / "models")
        },
        "loop_strategy": {
            "replay_buffer_size": 500,
            "checkpoint_interval": 10,
            "timeout_seconds": 3600
        }
    }

    # Test maximum speed (tradeoff=1)
    speed_intent = {"target_material": "Fe", "accuracy_speed_tradeoff": 1}
    speed_dict = config_dict.copy()
    speed_dict["intent"] = speed_intent
    config_speed = ProjectConfig.model_validate(speed_dict)

    # 0.15 - (1 * 0.013) = 0.137
    assert config_speed.distillation_config.uncertainty_threshold == pytest.approx(0.137)
    assert config_speed.loop_strategy.replay_buffer_size == 100

    # Test maximum accuracy (tradeoff=10)
    acc_intent = {"target_material": "Fe", "accuracy_speed_tradeoff": 10}
    acc_dict = config_dict.copy()
    acc_dict["intent"] = acc_intent
    config_acc = ProjectConfig.model_validate(acc_dict)

    # 0.15 - (10 * 0.013) = 0.02
    assert config_acc.distillation_config.uncertainty_threshold == pytest.approx(0.02)
    assert config_acc.loop_strategy.replay_buffer_size == 1000

    # Test backward compatibility (no intent)
    config_base = ProjectConfig.model_validate(config_dict)
    assert config_base.distillation_config.uncertainty_threshold == 0.05  # The default
    assert config_base.loop_strategy.replay_buffer_size == 500
