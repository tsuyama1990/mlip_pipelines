import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_submit_config_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient")
    monkeypatch.setattr("os.access", lambda x, y: True)

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True)
    proj_dir = tmp_dir / "myproj_api_success"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "README.md").touch()

    payload = {
        "project_root": str(proj_dir),
        "system": {"elements": ["Fe"]},
        "dynamics": {
            "trusted_directories": [],
            "project_root": str(proj_dir)
        },
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {
            "temp_dir": str(tmp_dir),
            "output_dir": str(tmp_dir),
            "model_storage_path": str(tmp_dir)
        },
        "loop_strategy": {
            "replay_buffer_size": 500,
            "checkpoint_interval": 5,
            "timeout_seconds": 86400
        },
        "intent": {
            "target_material": "Fe",
            "accuracy_speed_tradeoff": 10,
            "enable_auto_hpo": False
        }
    }

    response = client.post("/config/submit", json=payload)
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["message"] == "Configuration successfully validated"
    # Verify tradeoffs applied
    assert abs(data["config"]["distillation_config"]["uncertainty_threshold"] - 0.02) < 1e-6
    assert data["config"]["loop_strategy"]["replay_buffer_size"] == 1000

def test_submit_config_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient")
    monkeypatch.setattr("os.access", lambda x, y: True)

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True)
    proj_dir = tmp_dir / "myproj_api_error"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "README.md").touch()

    payload = {
        "project_root": str(proj_dir),
        "system": {"elements": ["Fe"]},
        "dynamics": {
            "trusted_directories": [],
            "project_root": str(proj_dir)
        },
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {
            "temp_dir": str(tmp_dir),
            "output_dir": str(tmp_dir),
            "model_storage_path": str(tmp_dir)
        },
        "loop_strategy": {
            "replay_buffer_size": 500,
            "checkpoint_interval": 5,
            "timeout_seconds": 86400
        },
        "intent": {
            "target_material": "../../etc/passwd",
            "accuracy_speed_tradeoff": 10,
            "enable_auto_hpo": False
        }
    }

    response = client.post("/config/submit", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
