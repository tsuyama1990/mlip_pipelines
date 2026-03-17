from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_validators(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.dynamics.security_utils.validate_executable_path",
        lambda *args, **kwargs: Path("/bin/sh"),
    )
    monkeypatch.setattr(
        "src.domain_models.config._check_allowed_base_dirs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("pathlib.Path.is_absolute", lambda self: True)
    monkeypatch.setattr("os.getuid", lambda: Path(tmp_path).stat().st_uid)


def test_submit_config_valid(tmp_path: Path) -> None:
    (tmp_path / "README.md").touch()
    payload = {
        "project_root": str(tmp_path),
        "system": {
            "elements": ["Fe", "Pt"],
            "baseline_potential": "zbl",
        },
        "dynamics": {
            "project_root": str(tmp_path),
            "trusted_directories": [str(tmp_path)],
        },
        "oracle": {},
        "trainer": {
            "trusted_directories": [str(tmp_path)],
        },
        "validator": {},
        "distillation_config": {
            "temp_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "model_storage_path": str(tmp_path),
        },
        "loop_strategy": {
            "replay_buffer_size": 100,
            "checkpoint_interval": 10,
            "timeout_seconds": 3600,
        },
        "intent": {"target_material": "Pt", "accuracy_speed_tradeoff": 1, "enable_auto_hpo": False},
    }

    response = client.post("/config/submit", json=payload)
    assert response.status_code == 200, response.json()

    response_data = response.json()
    assert response_data["distillation_config"]["uncertainty_threshold"] == 0.15
    assert response_data["loop_strategy"]["replay_buffer_size"] == 50


def test_submit_config_security_rejection(tmp_path: Path) -> None:
    payload = {
        "project_root": str(tmp_path),
        "system": {"elements": ["Fe", "Pt"], "baseline_potential": "zbl"},
        "dynamics": {"project_root": str(tmp_path), "trusted_directories": [str(tmp_path)]},
        "oracle": {},
        "trainer": {"trusted_directories": [str(tmp_path)]},
        "validator": {},
        "distillation_config": {
            "temp_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "model_storage_path": str(tmp_path),
        },
        "loop_strategy": {
            "replay_buffer_size": 100,
            "checkpoint_interval": 10,
            "timeout_seconds": 3600,
        },
        "intent": {
            "target_material": "../../etc/passwd",
            "accuracy_speed_tradeoff": 1,
            "enable_auto_hpo": False,
        },
    }

    response = client.post("/config/submit", json=payload)
    assert response.status_code == 422
    assert "target_material" in str(response.json())
