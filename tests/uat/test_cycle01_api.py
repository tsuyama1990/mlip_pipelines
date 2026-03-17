import os
import sys

# Ensure local imports work during direct script execution
sys.path.insert(0, os.getcwd())

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

@app.cell
def __() -> tuple:
    import json
    import tempfile
    from pathlib import Path

    from fastapi.testclient import TestClient
    from src import app

    client = TestClient(app)

    tmp_dir = Path(tempfile.mkdtemp()).resolve(strict=True)
    (tmp_dir / "README.md").touch()

    # Scenario ID: UAT-01-01: Intent-Driven Translation Validation
    print("Executing Scenario ID: UAT-01-01: Intent-Driven Translation Validation")
    payload_success = {
        "project_root": str(tmp_dir),
        "system": {
            "elements": ["Fe", "Pt"]
        },
        "dynamics": {
            "project_root": str(tmp_dir),
            "trusted_directories": []
        },
        "oracle": {},
        "trainer": {
            "trusted_directories": []
        },
        "validator": {},
        "distillation_config": {
            "temp_dir": str(tmp_dir),
            "output_dir": str(tmp_dir),
            "model_storage_path": str(tmp_dir)
        },
        "loop_strategy": {
            "replay_buffer_size": 500,
            "checkpoint_interval": 5,
            "timeout_seconds": 3600
        },
        "intent": {
            "target_material": "FePt",
            "accuracy_speed_tradeoff": 1,
            "enable_auto_hpo": False
        }
    }

    response_success = client.post("/config/submit", json=payload_success)
    assert response_success.status_code == 200, f"Expected 200 OK, got {response_success.status_code}"

    data_success = response_success.json()
    assert abs(data_success["distillation_config"]["uncertainty_threshold"] - 0.137) < 1e-5
    assert data_success["loop_strategy"]["replay_buffer_size"] == 100
    print("✓ UAT-01-01 Passed: Validated exact parameter mapping from high-level tradeoff intent.")

    # Scenario ID: UAT-01-02: Strict Security Validation of GUI Payloads
    print("Executing Scenario ID: UAT-01-02: Strict Security Validation of GUI Payloads")
    payload_malicious = {**payload_success}
    payload_malicious["intent"] = {
        "target_material": "../../etc/passwd",
        "accuracy_speed_tradeoff": 1,
        "enable_auto_hpo": False
    }

    response_malicious = client.post("/config/submit", json=payload_malicious)
    assert response_malicious.status_code == 422, f"Expected 422, got {response_malicious.status_code}"
    assert "Path traversal sequences" in response_malicious.text
    print("✓ UAT-01-02 Passed: Malicious payloads rejected accurately at the gateway.")

    # Scenario ID: UAT-01-03: Backward Compatibility with CLI Workflows
    print("Executing Scenario ID: UAT-01-03: Backward Compatibility with CLI Workflows")
    payload_expert = {**payload_success}
    payload_expert.pop("intent")
    # Setting an explicit expert parameter
    payload_expert["distillation_config"]["uncertainty_threshold"] = 0.05

    response_expert = client.post("/config/submit", json=payload_expert)
    assert response_expert.status_code == 200, f"Expected 200 OK, got {response_expert.status_code}"

    data_expert = response_expert.json()
    assert data_expert["intent"] is None
    assert abs(data_expert["distillation_config"]["uncertainty_threshold"] - 0.05) < 1e-5
    print("✓ UAT-01-03 Passed: CLI/Expert workflows are untouched and preserved perfectly.")

    return client, data_expert, data_success, json, Path, tempfile, tmp_dir

if __name__ == "__main__":
    app.run()
