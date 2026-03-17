import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_submit_config_intent_translation():
    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True) / "myproj_api_uat"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "README.md").touch()

    payload = {
        "project_root": str(tmp_dir),
        "system": {"elements": ["Fe", "O"]},
        "dynamics": {
            "project_root": str(tmp_dir),
            "trusted_directories": [],
            "lmp_binary": "lmp",
            "eon_binary": "eonclient"
        },
        "oracle": {},
        "trainer": {"trusted_directories": [], "pace_train_binary": "pace_train", "pace_activeset_binary": "pace_activeset", "mace_train_binary": "mace_run_train"},
        "validator": {},
        "distillation_config": {"temp_dir": str(tmp_dir / "tmp"), "output_dir": str(tmp_dir / "out"), "model_storage_path": str(tmp_dir / "store")},
        "loop_strategy": {"replay_buffer_size": 0, "checkpoint_interval": 10, "timeout_seconds": 3600},
        "intent": {
            "target_material": "PtNi",
            "accuracy_speed_tradeoff": 1,
            "enable_auto_hpo": False
        }
    }

    # We must patch shutil.which to not fail validation for lmp and eonclient
    import unittest.mock
    with unittest.mock.patch("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"):
        with unittest.mock.patch("os.access", return_value=True):
            response = client.post("/config/submit", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["distillation_config"]["uncertainty_threshold"] == 0.15
    assert data["loop_strategy"]["replay_buffer_size"] == 100

    # Test accuracy end
    payload["intent"]["accuracy_speed_tradeoff"] = 10
    with unittest.mock.patch("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"):
        with unittest.mock.patch("os.access", return_value=True):
            response = client.post("/config/submit", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["distillation_config"]["uncertainty_threshold"] == 0.02
    assert data["loop_strategy"]["replay_buffer_size"] == 1000

def test_submit_config_security_validation():
    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True) / "myproj_api_uat"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "README.md").touch()

    payload = {
        "project_root": str(tmp_dir),
        "system": {"elements": ["Fe", "O"]},
        "dynamics": {
            "project_root": str(tmp_dir),
            "trusted_directories": [],
            "lmp_binary": "lmp",
            "eon_binary": "eonclient"
        },
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {"temp_dir": str(tmp_dir / "tmp"), "output_dir": str(tmp_dir / "out"), "model_storage_path": str(tmp_dir / "store")},
        "loop_strategy": {"replay_buffer_size": 0, "checkpoint_interval": 10, "timeout_seconds": 3600},
        "intent": {
            "target_material": "../../etc/passwd",
            "accuracy_speed_tradeoff": 1,
            "enable_auto_hpo": False
        }
    }

    import unittest.mock
    with unittest.mock.patch("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"):
        with unittest.mock.patch("os.access", return_value=True):
            response = client.post("/config/submit", json=payload)

    assert response.status_code == 422
