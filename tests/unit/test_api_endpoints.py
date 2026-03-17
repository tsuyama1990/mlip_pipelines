# ruff: noqa: S103, PTH118, PTH123, PTH101
import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def valid_project_config_payload():
    temp_dir = tempfile.mkdtemp()

    trusted_bin = os.path.join(temp_dir, "lmp")
    with open(trusted_bin, "w") as f:
        f.write("#!/bin/sh\nexit 0")
    os.chmod(trusted_bin, 0o755)

    pacemaker_bin = os.path.join(temp_dir, "pacemaker")
    with open(pacemaker_bin, "w") as f:
        f.write("#!/bin/sh\nexit 0")
    os.chmod(pacemaker_bin, 0o755)

    mace_bin = os.path.join(temp_dir, "mace")
    with open(mace_bin, "w") as f:
        f.write("#!/bin/sh\nexit 0")
    os.chmod(mace_bin, 0o755)

    with open(os.path.join(temp_dir, "pyproject.toml"), "w") as f:
        f.write("[project]\nname='test'\n")

    yield {
        "project_root": temp_dir,
        "system": {
            "elements": ["Pt", "Ni"],
            "restricted_directories": ["/etc", "/bin", "/sbin", "/usr", "/var", "/root"],
        },
        "dynamics": {"project_root": temp_dir, "trusted_directories": [temp_dir]},
        "oracle": {},
        "trainer": {"trusted_directories": [temp_dir]},
        "validator": {},
        "distillation_config": {
            "enable": True,
            "mace_model_path": "mace-mp-0-medium",
            "uncertainty_threshold": 0.05,
            "sampling_structures_per_system": 100,
            "device": "cpu",
            "default_dtype": "float32",
            "dispersion": False,
            "temp_dir": os.path.join(temp_dir, "dist"),
            "output_dir": os.path.join(temp_dir, "dist_out"),
            "model_storage_path": os.path.join(temp_dir, "dist_store"),
        },
        "loop_strategy": {
            "use_tiered_oracle": True,
            "incremental_update": True,
            "replay_buffer_size": 1000,
            "baseline_potential_type": "LJ",
            "checkpoint_interval": 5,
            "max_retries": 3,
            "timeout_seconds": 3600,
        },
        "intent": {
            "target_material": "Pt-Ni",
            "accuracy_speed_tradeoff": 1,
            "enable_auto_hpo": False,
        },
    }

    shutil.rmtree(temp_dir)


def test_api_submit_config_valid_tradeoff_1(valid_project_config_payload):
    response = client.post("/config/submit", json=valid_project_config_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["intent"]["accuracy_speed_tradeoff"] == 1
    assert data["distillation_config"]["uncertainty_threshold"] == 0.15
    assert data["loop_strategy"]["replay_buffer_size"] == 500


def test_api_submit_config_valid_tradeoff_10(valid_project_config_payload):
    valid_project_config_payload["intent"]["accuracy_speed_tradeoff"] = 10
    response = client.post("/config/submit", json=valid_project_config_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["intent"]["accuracy_speed_tradeoff"] == 10
    assert data["distillation_config"]["uncertainty_threshold"] == 0.02
    assert data["loop_strategy"]["replay_buffer_size"] == 5000


def test_api_submit_config_invalid_tradeoff(valid_project_config_payload):
    valid_project_config_payload["intent"]["accuracy_speed_tradeoff"] = 11
    response = client.post("/config/submit", json=valid_project_config_payload)
    assert response.status_code == 422


def test_api_submit_config_invalid_string(valid_project_config_payload):
    valid_project_config_payload["intent"]["target_material"] = "../../etc/passwd"
    response = client.post("/config/submit", json=valid_project_config_payload)
    assert response.status_code == 422
