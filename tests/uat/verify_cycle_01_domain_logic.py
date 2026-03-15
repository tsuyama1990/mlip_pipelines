from pathlib import Path

import pytest
from pydantic import ValidationError


def test_uat_c01_01_threshold_logic():
    from src.domain_models.config import ActiveLearningThresholds

    with pytest.raises(ValidationError) as exc_info:
        ActiveLearningThresholds(threshold_call_dft=0.01, threshold_add_train=0.05)
    assert "must be strictly greater than or equal to local training addition threshold" in str(exc_info.value)

def test_uat_c01_02_cutout_constraints():
    from src.domain_models.config import CutoutConfig

    with pytest.raises(ValidationError) as exc_info:
        CutoutConfig(core_radius=6.0, buffer_radius=4.0)
    assert "must be strictly greater than core radius" in str(exc_info.value)

def test_uat_c01_03_legacy_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.domain_models.config import (
        DynamicsConfig,
        OracleConfig,
        ProjectConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )

    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient")
    monkeypatch.setattr("os.access", lambda x, y: True)

    proj_dir = tmp_path / "myproj_legacy"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "README.md").touch()

    # Missing completely the CutoutConfig, DistillationConfig, LoopStrategyConfig blocks
    config = ProjectConfig(
        project_root=proj_dir,
        system=SystemConfig(elements=["Fe", "O"]),
        dynamics=DynamicsConfig(project_root=str(proj_dir), trusted_directories=[]),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig()
    )

    assert config.cutout_config.core_radius == 3.0
    assert config.cutout_config.buffer_radius == 4.0
    assert config.cutout_config.enable_passivation is True
    assert config.distillation_config.enable is True

def test_uat_c01_04_distillation_overrides():
    from src.domain_models.config import DistillationConfig

    config = DistillationConfig(
        mace_model_path="my-custom-model.pt",
        sampling_structures_per_system=5000
    )

    assert config.mace_model_path == "my-custom-model.pt"
    assert config.sampling_structures_per_system == 5000

    with pytest.raises(ValidationError) as exc_info:
        DistillationConfig(sampling_structures_per_system=-100)
    assert "must be an integer strictly greater than zero" in str(exc_info.value)

def test_uat_c01_05_unexpected_fields():
    from src.domain_models.config import ActiveLearningThresholds

    with pytest.raises(ValidationError) as exc_info:
        ActiveLearningThresholds(invalid_threshold_parameter=0.05) # type: ignore[call-arg]
    assert "Extra inputs are not permitted" in str(exc_info.value)
