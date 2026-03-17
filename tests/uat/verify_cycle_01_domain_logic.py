import os
import shutil
import tempfile

from fastapi.testclient import TestClient


def main():
    # Insert cwd to sys.path to allow imports from src
    import sys

    sys.path.insert(0, os.getcwd())

    from src.api.main import app
    from src.domain_models.config import ProjectConfig

    print("Running CYCLE01 UAT: Intent-Driven Translation Validation...")
    client = TestClient(app)

    # Prepare temp workspace
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

    base_payload = {
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
    }

    # Scenario UAT-01-01: Intent-Driven Translation Validation
    print("Testing UAT-01-01: Intent-Driven Translation Validation (Speed = 1)")
    payload_intent = base_payload.copy()
    payload_intent["intent"] = {
        "target_material": "Pt-Ni",
        "accuracy_speed_tradeoff": 1,
        "enable_auto_hpo": False,
    }

    response = client.post("/config/submit", json=payload_intent)
    assert response.status_code == 200, f"UAT-01-01 failed: {response.text}"
    data = response.json()
    assert data["distillation_config"]["uncertainty_threshold"] == 0.15
    assert data["loop_strategy"]["replay_buffer_size"] == 500
    print("UAT-01-01 passed.")

    # Scenario UAT-01-02: Strict Security Validation of GUI Payloads
    print("Testing UAT-01-02: Strict Security Validation of GUI Payloads")
    payload_malicious = base_payload.copy()
    payload_malicious["intent"] = {
        "target_material": "../../etc/passwd",
        "accuracy_speed_tradeoff": 5,
        "enable_auto_hpo": False,
    }
    response_malicious = client.post("/config/submit", json=payload_malicious)
    assert response_malicious.status_code == 422
    assert "Path traversal or directory characters are not allowed" in response_malicious.text
    print("UAT-01-02 passed.")

    # Scenario UAT-01-03: Backward Compatibility with CLI Workflows
    print("Testing UAT-01-03: Backward Compatibility with CLI Workflows")
    payload_no_intent = base_payload.copy()
    # explicitly parse it via python Model directly instead of FastAPI to mimic CLI load
    config_obj = ProjectConfig.model_validate(payload_no_intent)
    assert config_obj.intent is None
    assert config_obj.distillation_config.uncertainty_threshold == 0.05  # untouched
    assert config_obj.loop_strategy.replay_buffer_size == 1000  # untouched
    print("UAT-01-03 passed.")

    shutil.rmtree(temp_dir)
    print("All CYCLE01 UAT passed successfully!")


if __name__ == "__main__":
    main()
