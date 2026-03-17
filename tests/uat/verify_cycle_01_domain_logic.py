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
            config = ProjectConfig(
                project_root=tmp_dir,
                system=SystemConfig(elements=["Fe", "O"]),
                dynamics=DynamicsConfig(project_root=str(tmp_dir), trusted_directories=[]),
                oracle=OracleConfig(),
                trainer=TrainerConfig(trusted_directories=[]),
                validator=ValidatorConfig(),
                distillation_config={"temp_dir": str(tmp_dir), "output_dir": str(tmp_dir), "model_storage_path": str(tmp_dir)},
                loop_strategy={"replay_buffer_size": 500, "checkpoint_interval": 10, "timeout_seconds": 3600}
            )

    assert config.cutout_config.core_radius == 3.0
    assert config.cutout_config.buffer_radius == 4.0
    assert config.cutout_config.enable_passivation is True
    assert config.distillation_config.enable is True


def test_uat_c01_04_distillation_overrides():
    from src.domain_models.config import DistillationConfig

    config = DistillationConfig(
        mace_model_path="mace-mp-0-large", sampling_structures_per_system=5000,
        temp_dir="/tmp/work", output_dir="/tmp/out", model_storage_path="/tmp/models"
    )

    assert config.mace_model_path == "mace-mp-0-large"
    assert config.sampling_structures_per_system == 5000

    with pytest.raises(ValidationError) as exc_info:
        DistillationConfig(sampling_structures_per_system=-100, temp_dir="/tmp/work", output_dir="/tmp/out", model_storage_path="/tmp/models")
    assert "must be an integer strictly greater than zero" in str(exc_info.value)


def test_uat_c01_05_unexpected_fields():
    from src.domain_models.config import ActiveLearningThresholds

    with pytest.raises(ValidationError) as exc_info:
        ActiveLearningThresholds(invalid_threshold_parameter=0.05)  # type: ignore[call-arg]
    assert "Extra inputs are not permitted" in str(exc_info.value)


# Cycle 01 additions:
def test_uat_c01_06_intent_driven_translation_validation():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "pyproject.toml").touch()

        payload = {
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
            },
            "intent": {
                "target_material": "Pt-Ni",
                "accuracy_speed_tradeoff": 1
            }
        }

        response = client.post("/config/submit", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["config"]["distillation_config"]["uncertainty_threshold"] == 0.15
        assert data["config"]["loop_strategy"]["replay_buffer_size"] == 100


def test_uat_c01_07_strict_security_validation_of_gui_payloads():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "pyproject.toml").touch()

        payload = {
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
            },
            "intent": {
                "target_material": "../../etc/passwd",
                "accuracy_speed_tradeoff": 5
            }
        }

        response = client.post("/config/submit", json=payload)
        assert response.status_code == 422
        assert "Path traversal characters are not allowed" in str(response.json())


def test_uat_c01_08_backward_compatibility_with_cli_workflows():
    from src.domain_models.config import ProjectConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "pyproject.toml").touch()

        payload = {
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

        config = ProjectConfig(**payload)

        assert config.distillation_config.uncertainty_threshold == 0.05
        assert config.loop_strategy.replay_buffer_size == 500
