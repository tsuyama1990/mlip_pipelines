import marimo

__generated_with = "0.2.13"
app = marimo.App()


@app.cell
def __() -> tuple:
    import os
    import sys
    import tempfile
    import unittest.mock
    from pathlib import Path

    import pytest
    from pydantic import ValidationError

    # Inject project root BEFORE imports
    sys.path.insert(0, str(Path.cwd()))

    from src.domain_models.config import ProjectConfig

    return sys, os, tempfile, unittest, Path, pytest, ValidationError, ProjectConfig


@app.cell
def __(tempfile, Path, ProjectConfig, unittest) -> tuple:
    print("Executing UAT-01-01: Intent-Driven Translation Validation...")

    base_dir = Path(tempfile.mkdtemp())
    (base_dir / "pyproject.toml").touch()

    gui_payload = {
        "project_root": str(base_dir),
        "system": {"elements": ["Fe"]},
        "dynamics": {"project_root": str(base_dir), "trusted_directories": []},
        "oracle": {},
        "trainer": {"trusted_directories": []},
        "validator": {},
        "distillation_config": {
            "mace_model_path": "mace-mp-0-medium",
            "temp_dir": str(base_dir),
            "output_dir": str(base_dir),
            "model_storage_path": str(base_dir),
            "uncertainty_threshold": 0.05,
        },
        "loop_strategy": {
            "replay_buffer_size": 1000,
            "checkpoint_interval": 10,
            "timeout_seconds": 3600,
            "max_iterations": 20,
        },
        "intent": {"target_material": "Fe", "accuracy_speed_tradeoff": 1, "enable_auto_hpo": True},
    }

    with unittest.mock.patch(
        "src.domain_models.config._validate_env_file_security", return_value=Path("/tmp/.env")
    ):
        with unittest.mock.patch("dotenv.dotenv_values", return_value={}):
            with unittest.mock.patch("shutil.which", return_value="/usr/bin/mock_bin"):
                with unittest.mock.patch.object(
                    ProjectConfig, "validate_env_content", side_effect=lambda x: x
                ):
                    with unittest.mock.patch("os.access", return_value=True):
                        with unittest.mock.patch("pathlib.Path.exists", return_value=True):
                            with unittest.mock.patch("pathlib.Path.is_dir", return_value=True):
                                _config = ProjectConfig.model_validate(gui_payload)

    import math

    assert math.isclose(_config.distillation_config.uncertainty_threshold, 0.137, rel_tol=1e-5), (
        f"Threshold translation failed, got {_config.distillation_config.uncertainty_threshold}"
    )
    assert _config.loop_strategy.max_iterations == 10, "Max iterations translation failed"
    print("✓ UAT-01-01 Passed: Intent properly translated to thresholds and scaling parameters")
    return base_dir, gui_payload


@app.cell
def __(base_dir, gui_payload, Path, ProjectConfig, pytest, ValidationError, unittest) -> tuple:
    print("Executing UAT-01-02: Strict Security Validation of GUI Payloads...")

    malicious_payload = gui_payload.copy()
    malicious_payload["intent"] = {
        "target_material": "../../etc/passwd",
        "accuracy_speed_tradeoff": 5,
    }

    with unittest.mock.patch(
        "src.domain_models.config._validate_env_file_security", return_value=Path("/tmp/.env")
    ):
        with unittest.mock.patch("dotenv.dotenv_values", return_value={}):
            with unittest.mock.patch("shutil.which", return_value="/usr/bin/mock_bin"):
                with unittest.mock.patch.object(
                    ProjectConfig, "validate_env_content", side_effect=lambda x: x
                ):
                    with unittest.mock.patch("os.access", return_value=True):
                        with unittest.mock.patch("pathlib.Path.exists", return_value=True):
                            with unittest.mock.patch("pathlib.Path.is_dir", return_value=True):
                                with pytest.raises(ValidationError) as excinfo:
                                    ProjectConfig.model_validate(malicious_payload)
                    assert "Path traversal characters are not allowed" in str(excinfo.value), (
                        "Security validation failed to catch injection"
                    )

    print("✓ UAT-01-02 Passed: System properly rejected malicious path traversal payload")
    return (malicious_payload,)


@app.cell
def __(base_dir, gui_payload, Path, ProjectConfig, unittest) -> tuple:
    print("Executing UAT-01-03: Backward Compatibility with CLI Workflows...")

    legacy_payload = gui_payload.copy()
    del legacy_payload["intent"]

    with unittest.mock.patch(
        "src.domain_models.config._validate_env_file_security", return_value=Path("/tmp/.env")
    ):
        with unittest.mock.patch("dotenv.dotenv_values", return_value={}):
            with unittest.mock.patch("shutil.which", return_value="/usr/bin/mock_bin"):
                with unittest.mock.patch.object(
                    ProjectConfig, "validate_env_content", side_effect=lambda x: x
                ):
                    with unittest.mock.patch("os.access", return_value=True):
                        with unittest.mock.patch("pathlib.Path.exists", return_value=True):
                            with unittest.mock.patch("pathlib.Path.is_dir", return_value=True):
                                _config2 = ProjectConfig.model_validate(legacy_payload)

    assert _config2.distillation_config.uncertainty_threshold == 0.05, (
        "Legacy parameter altered incorrectly"
    )
    assert _config2.loop_strategy.max_iterations == 20, "Legacy parameter altered incorrectly"

    print("✓ UAT-01-03 Passed: CLI config parsed natively without intent translation")
    return (legacy_payload,)


if __name__ == "__main__":
    app.run()
