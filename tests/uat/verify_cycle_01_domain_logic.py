"""
UAT Scenario ID: UAT-01-01, UAT-01-02, UAT-01-03
Verification for Intent-Driven Translation, Strict Security, and CLI Backward Compatibility
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.domain_models.config import ProjectConfig

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


def test_uat_01_01_intent_translation(tmp_path: Path) -> None:
    """UAT-01-01: Intent-Driven Translation Validation."""
    (tmp_path / "README.md").touch()
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
        "intent": {"target_material": "Pt", "accuracy_speed_tradeoff": 1, "enable_auto_hpo": False},
    }
    response = client.post("/config/submit", json=payload)
    assert response.status_code == 200, response.text
    config = response.json()
    assert config["distillation_config"]["uncertainty_threshold"] == 0.15
    assert config["loop_strategy"]["replay_buffer_size"] == 50


def test_uat_01_02_strict_security(tmp_path: Path) -> None:
    """UAT-01-02: Strict Security Validation of GUI Payloads."""
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
    assert "target_material" in response.text

    # Check out-of-bounds integer
    payload["intent"] = {
        "target_material": "Pt",
        "accuracy_speed_tradeoff": 11,
        "enable_auto_hpo": False,
    }
    response2 = client.post("/config/submit", json=payload)
    assert response2.status_code == 422
    assert "accuracy_speed_tradeoff" in response2.text


def test_uat_01_03_backward_compatibility(tmp_path: Path) -> None:
    """UAT-01-03: Backward Compatibility with CLI Workflows."""
    (tmp_path / "README.md").touch()
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
            "uncertainty_threshold": 0.08,  # explicit setting
        },
        "loop_strategy": {
            "replay_buffer_size": 333,  # explicit setting
            "checkpoint_interval": 10,
            "timeout_seconds": 3600,
        },
    }

    # Directly parse using model_validate
    config = ProjectConfig.model_validate(payload)

    assert config.distillation_config.uncertainty_threshold == 0.08
    assert config.loop_strategy.replay_buffer_size == 333
