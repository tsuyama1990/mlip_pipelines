import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from src.api.main import app

client = TestClient(app)

def _create_base_project_config(tmp_path: Path) -> dict:
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

def test_config_submit_success(tmp_path):
    payload = _create_base_project_config(tmp_path)
    payload["intent"] = {
        "target_material": "Fe",
        "accuracy_speed_tradeoff": 1
    }

    response = client.post("/config/submit", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Verify the math translation happened because we passed it through the API
    assert data["config"]["distillation_config"]["uncertainty_threshold"] == 0.150
    assert data["config"]["loop_strategy"]["replay_buffer_size"] == 100


def test_config_submit_malicious_intent(tmp_path):
    payload = _create_base_project_config(tmp_path)
    payload["intent"] = {
        "target_material": "../../etc/passwd",
        "accuracy_speed_tradeoff": 5
    }

    response = client.post("/config/submit", json=payload)

    # Should be unprocessable entity
    assert response.status_code == 422
    assert "Path traversal characters are not allowed" in str(response.json())


def test_config_submit_backward_compatibility(tmp_path):
    # Missing intent entirely
    payload = _create_base_project_config(tmp_path)

    response = client.post("/config/submit", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["config"]["intent"] is None

    # Verify original values are kept
    assert data["config"]["distillation_config"]["uncertainty_threshold"] == 0.05
    assert data["config"]["loop_strategy"]["replay_buffer_size"] == 500
