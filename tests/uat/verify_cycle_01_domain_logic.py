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

    from src.domain_models.config import DistillationConfig, LoopStrategyConfig

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
                distillation_config=DistillationConfig(temp_dir=str(tmp_dir), output_dir=str(tmp_dir), model_storage_path=str(tmp_dir)),
                loop_strategy=LoopStrategyConfig(replay_buffer_size=500, checkpoint_interval=10, timeout_seconds=3600)
            )

    assert config.cutout_config.core_radius == 3.0
    assert config.cutout_config.buffer_radius == 4.0
    assert config.cutout_config.enable_passivation is True
    assert config.distillation_config.enable is True


def test_uat_c01_04_distillation_overrides():
    from src.domain_models.config import DistillationConfig

    config = DistillationConfig(
        mace_model_path="mace-mp-0-large", sampling_structures_per_system=5000,
        temp_dir="/tmp/a", output_dir="/tmp/b", model_storage_path="/tmp/c"
    )

    assert config.mace_model_path == "mace-mp-0-large"
    assert config.sampling_structures_per_system == 5000

    with pytest.raises(ValidationError) as exc_info:
        DistillationConfig(sampling_structures_per_system=-100)
    assert "must be an integer strictly greater than zero" in str(exc_info.value)


def test_uat_c01_05_unexpected_fields():
    from src.domain_models.config import ActiveLearningThresholds

    with pytest.raises(ValidationError) as exc_info:
        ActiveLearningThresholds(invalid_threshold_parameter=0.05)  # type: ignore[call-arg]
    assert "Extra inputs are not permitted" in str(exc_info.value)

def create_base_config_dict(tmp_path: Path) -> dict:
    (tmp_path / "pyproject.toml").touch()
    return {
        "project_root": str(tmp_path),
        "system": {"elements": ["Fe", "Pt"]},
        "dynamics": {"trusted_directories": [str(tmp_path)], "project_root": str(tmp_path)},
        "oracle": {},
        "trainer": {"trusted_directories": [str(tmp_path)]},
        "validator": {},
        "distillation_config": {
            "temp_dir": str(tmp_path / "temp"),
            "output_dir": str(tmp_path / "out"),
            "model_storage_path": str(tmp_path / "models")
        },
        "loop_strategy": {
            "replay_buffer_size": 500,
            "checkpoint_interval": 10,
            "timeout_seconds": 3600
        }
    }


def test_uat_01_01_translation(tmp_path: Path) -> None:
    """
    Scenario ID: UAT-01-01: Intent-Driven Translation Validation
    Verify that the system correctly translates a high-level user intent (the "Accuracy vs. Speed" slider).
    """
    from src.domain_models.config import ProjectConfig
    config_dict = create_base_config_dict(tmp_path)
    # Simulate an incoming payload with tradeoff 1 (maximum speed)
    config_dict["intent"] = {"target_material": "Fe", "accuracy_speed_tradeoff": 1}

    config = ProjectConfig.model_validate(config_dict)

    # 0.15 - (1 * 0.013) = 0.137, which is a permissive high value as requested
    assert config.distillation_config.uncertainty_threshold == pytest.approx(0.137)

    # replay_buffer_size should be minimal (1 * 100 = 100)
    assert config.loop_strategy.replay_buffer_size == 100


def test_uat_01_02_security(tmp_path: Path) -> None:
    """
    Scenario ID: UAT-01-02: Strict Security Validation of GUI Payloads
    Verify that the backend strictly rejects malicious string inputs (path traversal) from GUI intent.
    """
    from src.domain_models.config import ProjectConfig
    config_dict = create_base_config_dict(tmp_path)
    config_dict["intent"] = {"target_material": "../../etc/passwd", "accuracy_speed_tradeoff": 5}

    with pytest.raises(ValidationError) as exc_info:
        ProjectConfig.model_validate(config_dict)

    # Ensure error detail explicitly calls out path traversal rejection
    assert "Path traversal characters" in str(exc_info.value)


def test_uat_01_03_backward_compatibility(tmp_path: Path) -> None:
    """
    Scenario ID: UAT-01-03: Backward Compatibility with CLI Workflows
    Verify that existing configurations that omit the intent object still parse normally
    and retain their explicitly defined values.
    """
    from src.domain_models.config import ProjectConfig
    config_dict = create_base_config_dict(tmp_path)
    # Explicitly set deep hyperparameters instead of relying on intent
    config_dict["distillation_config"]["uncertainty_threshold"] = 0.08
    config_dict["loop_strategy"]["replay_buffer_size"] = 1234

    # We omit "intent" entirely
    assert "intent" not in config_dict

    config = ProjectConfig.model_validate(config_dict)

    # Mathematical translation should NOT override the explicitly defined values
    assert config.distillation_config.uncertainty_threshold == 0.08
    assert config.loop_strategy.replay_buffer_size == 1234
    assert config.intent is None

    print("✓ UAT verification complete for CYCLE01")
