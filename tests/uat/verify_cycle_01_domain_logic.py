import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_uat_c01_01_threshold_logic():
    from src.domain_models.config import ActiveLearningThresholds

    with pytest.raises(ValidationError) as exc_info:
        ActiveLearningThresholds(threshold_call_dft=0.01, threshold_add_train=0.05)
    assert "must be strictly greater than or equal to local training addition threshold" in str(
        exc_info.value
    )


def test_uat_c01_02_cutout_constraints():
    from src.domain_models.config import CutoutConfig

    with pytest.raises(ValidationError) as exc_info:
        CutoutConfig(core_radius=6.0, buffer_radius=4.0)
    assert "must be strictly greater than core radius" in str(exc_info.value)


def test_uat_c01_03_legacy_config():
    import unittest.mock

    from src.domain_models.config import (
        DistillationConfig,
        DynamicsConfig,
        LoopStrategyConfig,
        OracleConfig,
        ProjectConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True) / "myproj_legacy_uat"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "README.md").touch()

    with unittest.mock.patch(
        "shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"
    ):
        with unittest.mock.patch("os.access", return_value=True):
            config = ProjectConfig(
                project_root=tmp_dir,
                system=SystemConfig(elements=["Fe", "O"]),
                dynamics=DynamicsConfig(project_root=str(tmp_dir), trusted_directories=[]),
                oracle=OracleConfig(),
                trainer=TrainerConfig(trusted_directories=[]),
                validator=ValidatorConfig(),
                distillation_config=DistillationConfig(temp_dir="/tmp", output_dir="/tmp", model_storage_path="/tmp"),
                loop_strategy=LoopStrategyConfig(replay_buffer_size=500, checkpoint_interval=5, timeout_seconds=86400),
            )

    assert config.cutout_config.core_radius == 3.0
    assert config.cutout_config.buffer_radius == 4.0
    assert config.cutout_config.enable_passivation is True
    assert config.distillation_config.enable is True


def test_uat_c01_04_distillation_overrides():
    from src.domain_models.config import DistillationConfig

    config = DistillationConfig(
        mace_model_path="mace-mp-0-large", sampling_structures_per_system=5000, temp_dir="/tmp", output_dir="/tmp", model_storage_path="/tmp"
    )

    assert config.mace_model_path == "mace-mp-0-large"
    assert config.sampling_structures_per_system == 5000

    with pytest.raises(ValidationError) as exc_info:
        DistillationConfig(sampling_structures_per_system=-100, temp_dir="/tmp", output_dir="/tmp", model_storage_path="/tmp")
    assert "must be an integer strictly greater than zero" in str(exc_info.value)


def test_uat_c01_05_unexpected_fields():
    from src.domain_models.config import ActiveLearningThresholds

    with pytest.raises(ValidationError) as exc_info:
        ActiveLearningThresholds(invalid_threshold_parameter=0.05)  # type: ignore[call-arg]
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_uat_c01_01_intent_translation(monkeypatch: pytest.MonkeyPatch):
    import json
    import tempfile

    from src.domain_models.config import ProjectConfig

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True) / "myproj_intent_uat"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "README.md").touch()

    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"
    )
    monkeypatch.setattr("os.access", lambda x, y: True)

    payload = {
        "project_root": str(tmp_dir),
        "system": {"elements": ["Pt", "Ni"]},
        "dynamics": {"project_root": str(tmp_dir), "trusted_directories": []},
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {"temp_dir": "/tmp", "output_dir": "/tmp", "model_storage_path": "/tmp"},
        "loop_strategy": {"replay_buffer_size": 500, "checkpoint_interval": 5, "timeout_seconds": 86400},
        "intent": {
            "target_material": "Pt-Ni",
            "accuracy_speed_tradeoff": 1
        }
    }

    config = ProjectConfig.model_validate_json(json.dumps(payload))
    assert config.distillation_config.uncertainty_threshold == pytest.approx(0.137)
    assert config.loop_strategy.replay_buffer_size == 100


def test_uat_c01_02_strict_security(monkeypatch: pytest.MonkeyPatch):
    import json
    import tempfile

    from src.domain_models.config import ProjectConfig

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True) / "myproj_intent_uat_sec"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "README.md").touch()

    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"
    )
    monkeypatch.setattr("os.access", lambda x, y: True)

    payload = {
        "project_root": str(tmp_dir),
        "system": {"elements": ["Pt", "Ni"]},
        "dynamics": {"project_root": str(tmp_dir), "trusted_directories": []},
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {"temp_dir": "/tmp", "output_dir": "/tmp", "model_storage_path": "/tmp"},
        "loop_strategy": {"replay_buffer_size": 500, "checkpoint_interval": 5, "timeout_seconds": 86400},
        "intent": {
            "target_material": "../../etc/passwd",
            "accuracy_speed_tradeoff": 1
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        ProjectConfig.model_validate_json(json.dumps(payload))

    assert "Path traversal sequences (..) are not allowed" in str(exc_info.value)


def test_uat_c01_03_backward_compatibility(monkeypatch: pytest.MonkeyPatch):
    import json
    import tempfile

    from src.domain_models.config import ProjectConfig

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True) / "myproj_intent_uat_bwd"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "README.md").touch()

    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"
    )
    monkeypatch.setattr("os.access", lambda x, y: True)

    payload = {
        "project_root": str(tmp_dir),
        "system": {"elements": ["Pt", "Ni"]},
        "dynamics": {"project_root": str(tmp_dir), "trusted_directories": []},
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {"temp_dir": "/tmp", "output_dir": "/tmp", "model_storage_path": "/tmp", "uncertainty_threshold": 0.05},
        "loop_strategy": {"replay_buffer_size": 500, "checkpoint_interval": 5, "timeout_seconds": 86400}
    }

    config = ProjectConfig.model_validate_json(json.dumps(payload))
    # It passes with defaults since intent is not present, no translation occurs
    assert config.distillation_config.uncertainty_threshold == 0.05
    assert config.loop_strategy.replay_buffer_size == 500
