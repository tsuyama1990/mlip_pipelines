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
        DynamicsConfig,
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
            from src.domain_models.config import DistillationConfig, LoopStrategyConfig
            config = ProjectConfig(
                project_root=tmp_dir,
                system=SystemConfig(elements=["Fe", "O"]),
                dynamics=DynamicsConfig(project_root=str(tmp_dir), trusted_directories=[]),
                oracle=OracleConfig(),
                trainer=TrainerConfig(trusted_directories=[]),
                validator=ValidatorConfig(),
                distillation_config=DistillationConfig(temp_dir="/tmp", output_dir="/tmp", model_storage_path="/tmp"),
                loop_strategy=LoopStrategyConfig(replay_buffer_size=500, checkpoint_interval=5, timeout_seconds=86400)
            )

    assert config.cutout_config.core_radius == 3.0
    assert config.cutout_config.buffer_radius == 4.0
    assert config.cutout_config.enable_passivation is True
    assert config.distillation_config.enable is True


def test_uat_c01_04_distillation_overrides():
    from src.domain_models.config import DistillationConfig

    config = DistillationConfig(
        mace_model_path="mace-mp-0-large", sampling_structures_per_system=5000,
        temp_dir="/tmp", output_dir="/tmp", model_storage_path="/tmp"
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

def test_uat_01_01_intent_driven_translation_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from src.api.main import app as fastapi_app

    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient")
    monkeypatch.setattr("os.access", lambda x, y: True)

    client = TestClient(fastapi_app)

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True)
    proj_dir = tmp_dir / "myproj_uat_01_01"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "README.md").touch()

    payload = {
        "project_root": str(proj_dir),
        "system": {"elements": ["Fe"]},
        "dynamics": {"trusted_directories": [], "project_root": str(proj_dir)},
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {"temp_dir": "/tmp", "output_dir": "/tmp", "model_storage_path": "/tmp"},
        "loop_strategy": {"replay_buffer_size": 500, "checkpoint_interval": 5, "timeout_seconds": 86400},
        "intent": {"target_material": "Fe", "accuracy_speed_tradeoff": 1, "enable_auto_hpo": False}
    }

    response = client.post("/config/submit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert abs(data["config"]["distillation_config"]["uncertainty_threshold"] - 0.137) < 1e-6
    assert data["config"]["loop_strategy"]["replay_buffer_size"] == 100

def test_uat_01_02_strict_security_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from src.api.main import app as fastapi_app

    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient")
    monkeypatch.setattr("os.access", lambda x, y: True)

    client = TestClient(fastapi_app)

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True)
    proj_dir = tmp_dir / "myproj_uat_01_02"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "README.md").touch()

    payload = {
        "project_root": str(proj_dir),
        "system": {"elements": ["Fe"]},
        "dynamics": {"trusted_directories": [], "project_root": str(proj_dir)},
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {"temp_dir": "/tmp", "output_dir": "/tmp", "model_storage_path": "/tmp"},
        "loop_strategy": {"replay_buffer_size": 500, "checkpoint_interval": 5, "timeout_seconds": 86400},
        "intent": {"target_material": "../../etc/passwd", "accuracy_speed_tradeoff": 5, "enable_auto_hpo": False}
    }

    response = client.post("/config/submit", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_uat_01_03_backward_compatibility() -> None:
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

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True) / "myproj_legacy_uat_03"
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
                loop_strategy=LoopStrategyConfig(replay_buffer_size=500, checkpoint_interval=5, timeout_seconds=86400)
            )

    assert config.intent is None
    assert config.loop_strategy.replay_buffer_size == 500
    assert config.distillation_config.uncertainty_threshold == 0.05
