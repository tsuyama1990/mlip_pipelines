from pathlib import Path

import pytest
from pydantic import ValidationError

from src.domain_models.config import (
    DynamicsConfig,
    OracleConfig,
    ProjectConfig,
    SystemConfig,
    TrainerConfig,
    ValidatorConfig,
)


def test_system_config_valid() -> None:
    config = SystemConfig(elements=["Fe", "Pt"])
    assert config.elements == ["Fe", "Pt"]
    assert config.baseline_potential == "zbl"


def test_system_config_invalid() -> None:
    with pytest.raises(ValidationError):
        SystemConfig(elements=[])  # Empty list violates min_length=1

    with pytest.raises(ValidationError):
        SystemConfig(elements=["Fe"], extra_field="bad")  # type: ignore[call-arg]


def test_dynamics_config_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"
    )
    monkeypatch.setattr("os.access", lambda x, y: True)

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True)
    proj_dir = tmp_dir / "myproj2"
    proj_dir.mkdir(parents=True, exist_ok=True)

    config = DynamicsConfig(
        uncertainty_threshold=10.0, project_root=str(proj_dir), trusted_directories=[]
    )
    assert config.uncertainty_threshold == 10.0


def test_oracle_config_invalid() -> None:
    with pytest.raises(ValidationError):
        OracleConfig(kspacing=-0.1)  # gt=0.0 constraint violated


def test_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    monkeypatch.setattr(
        "shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient"
    )
    monkeypatch.setattr("os.access", lambda x, y: True)

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True)
    proj_dir = tmp_dir / "myproj3"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "README.md").touch()

    config = ProjectConfig(
        project_root=proj_dir,
        system=SystemConfig(elements=["Fe", "O"]),
        dynamics=DynamicsConfig(project_root=str(proj_dir), trusted_directories=[]),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig(),
    )
    assert config.system.elements == ["Fe", "O"]
    assert config.system.baseline_potential == "zbl"


def test_validate_single_trusted_dir_valid(tmp_path: Path) -> None:
    d = tmp_path / "valid_dir"
    d.mkdir()
    from src.domain_models.config import (
        _secure_resolve_and_validate_dir as _validate_single_trusted_dir,
    )

    assert _validate_single_trusted_dir(str(d), check_exists=True) == str(d.resolve())


def test_validate_single_trusted_dir_not_exist(tmp_path: Path) -> None:
    d = tmp_path / "not_exist"
    from src.domain_models.config import (
        _secure_resolve_and_validate_dir as _validate_single_trusted_dir,
    )

    with pytest.raises(ValueError, match="Directory does not exist"):
        _validate_single_trusted_dir(str(d), check_exists=True)


def test_validate_single_trusted_dir_not_dir(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("dummy")
    from src.domain_models.config import (
        _secure_resolve_and_validate_dir as _validate_single_trusted_dir,
    )

    with pytest.raises(ValueError, match="(?i).*must be a directory.*"):
        _validate_single_trusted_dir(str(f))


def test_validate_kspacing() -> None:
    from pydantic import ValidationError

    from src.domain_models.config import OracleConfig

    with pytest.raises(ValidationError, match=".*kspacing.*"):
        OracleConfig(kspacing=0.0)


def test_project_config_env_key() -> None:
    from src.domain_models.config import _validate_env_key

    with pytest.raises(ValueError, match=".*Unauthorized environment variable injected via .env.*"):
        _validate_env_key("INVALID KEY")


def test_project_config_env_value() -> None:
    from src.domain_models.config import _validate_env_value

    with pytest.raises(ValueError, match=".*Invalid characters detected.*"):
        _validate_env_value("value; rm -rf")

    # "../secret" is now valid under the relaxed r"^[-a-zA-Z0-9_.:/=,+]*$" rule
    # and no longer triggers a ValueError in _validate_env_value itself
    _validate_env_value("../secret")


def test_project_config_env_file_security(tmp_path: Path) -> None:

    from src.domain_models.config import _validate_env_file_security

    base = tmp_path / "base"
    base.mkdir()
    env = base / ".env"

    # Test file doesn't exist
    with pytest.raises(FileNotFoundError):
        _validate_env_file_security(env, base)

    # Test symlink
    target = base / "target.txt"
    target.write_text("test")
    env.symlink_to(target)
    with pytest.raises(ValueError, match=".*must not be a symlink.*"):
        _validate_env_file_security(env, base)

    env.unlink()

    # Test oversized file
    with env.open("wb") as f:
        f.write(b"0" * (11 * 1024))

    with pytest.raises(ValueError, match=".*exceeds maximum allowed size.*"):
        _validate_env_file_security(env, base)

    env.unlink()

    # Test bad permissions (world writable)
    env.write_text("TEST=1")
    env.chmod(0o777)
    # The system might override 777 depending on umask, so ensure group write is on
    import stat

    if bool(env.stat().st_mode & stat.S_IWOTH):
        with pytest.raises(ValueError, match=".*insecure permissions.*"):
            _validate_env_file_security(env, base)


def test_project_config_convert_str_to_path() -> None:
    from pathlib import Path

    from src.domain_models.config import ProjectConfig

    res = ProjectConfig.convert_str_to_path("/")
    assert isinstance(res, Path)
    assert str(res) == "/"


def test_project_config_validate_project_root() -> None:
    import contextlib
    from pathlib import Path

    from src.domain_models.config import ProjectConfig

    with contextlib.suppress(FileNotFoundError):
        ProjectConfig.validate_project_root(Path("relative/path"))

def test_distillation_config_valid() -> None:
    from src.domain_models.config import DistillationConfig
    config = DistillationConfig(uncertainty_threshold=0.1, sampling_structures_per_system=500)
    assert config.uncertainty_threshold == 0.1
    assert config.sampling_structures_per_system == 500

def test_distillation_config_invalid() -> None:
    from pydantic import ValidationError

    from src.domain_models.config import DistillationConfig

    with pytest.raises(ValidationError, match="uncertainty_threshold must be strictly positive"):
        DistillationConfig(uncertainty_threshold=0.0)

    with pytest.raises(ValidationError, match="sampling_structures_per_system must be an integer strictly greater than zero"):
        DistillationConfig(sampling_structures_per_system=-10)

def test_active_learning_thresholds_valid() -> None:
    from src.domain_models.config import ActiveLearningThresholds
    config = ActiveLearningThresholds(threshold_call_dft=0.1, threshold_add_train=0.05, smooth_steps=5)
    assert config.threshold_call_dft == 0.1
    assert config.threshold_add_train == 0.05
    assert config.smooth_steps == 5

def test_active_learning_thresholds_invalid() -> None:
    from pydantic import ValidationError

    from src.domain_models.config import ActiveLearningThresholds

    with pytest.raises(ValidationError, match="must be strictly greater than or equal to"):
        ActiveLearningThresholds(threshold_call_dft=0.01, threshold_add_train=0.05)

    with pytest.raises(ValidationError, match="smooth_steps must be strictly greater than zero"):
        ActiveLearningThresholds(smooth_steps=0)

def test_cutout_config_valid() -> None:
    from src.domain_models.config import CutoutConfig
    config = CutoutConfig(core_radius=3.0, buffer_radius=4.0)
    assert config.core_radius == 3.0
    assert config.buffer_radius == 4.0

def test_cutout_config_invalid() -> None:
    from pydantic import ValidationError

    from src.domain_models.config import CutoutConfig

    with pytest.raises(ValidationError, match="core_radius must be strictly positive"):
        CutoutConfig(core_radius=0.0, buffer_radius=4.0)

    with pytest.raises(ValidationError, match="buffer_radius must be strictly positive"):
        CutoutConfig(core_radius=3.0, buffer_radius=-1.0)

    with pytest.raises(ValidationError, match="must be strictly greater than core radius"):
        CutoutConfig(core_radius=5.0, buffer_radius=3.0)

def test_project_config_legacy_compat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/lmp" if x == "lmp" else "/usr/bin/eonclient")
    monkeypatch.setattr("os.access", lambda x, y: True)

    tmp_dir = Path(tempfile.gettempdir()).resolve(strict=True)
    proj_dir = tmp_dir / "myproj4"
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "README.md").touch()

    # Missing new fields (legacy config equivalent)
    config = ProjectConfig(
        project_root=proj_dir,
        system=SystemConfig(elements=["Fe"]),
        dynamics=DynamicsConfig(project_root=str(proj_dir), trusted_directories=[]),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig()
    )

    # Assert defaults were correctly applied
    assert config.distillation_config.enable is True
    assert config.cutout_config.core_radius == 3.0
    assert config.loop_strategy.use_tiered_oracle is True

def test_extra_forbid() -> None:
    from pydantic import ValidationError

    from src.domain_models.config import CutoutConfig

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CutoutConfig(core_radius=4.0, buffer_radius=5.0, invalid_field=True) # type: ignore[call-arg]


def test_distillation_config_path_validation() -> None:
    from pydantic import ValidationError

    from src.domain_models.config import DistillationConfig

    # Valid string name
    config = DistillationConfig(mace_model_path="mace-mp-0-large")
    assert config.mace_model_path == "mace-mp-0-large"

    # Valid path
    config = DistillationConfig(mace_model_path="/absolute/path/to/my_model.pt")
    assert config.mace_model_path == "/absolute/path/to/my_model.pt"

    # Invalid string name
    with pytest.raises(ValidationError, match="Unknown model name or unsupported extension"):
        DistillationConfig(mace_model_path="unknown-model")

    # Invalid path traversal
    with pytest.raises(ValidationError, match="Path traversal sequences"):
        DistillationConfig(mace_model_path="../hidden_model.pt")

def test_loop_strategy_consistency() -> None:
    from pydantic import ValidationError

    from src.domain_models.config import LoopStrategyConfig

    config = LoopStrategyConfig(use_tiered_oracle=True, incremental_update=True)
    assert config.incremental_update is True

    with pytest.raises(ValidationError, match="incremental_update cannot be True when use_tiered_oracle is False"):
        LoopStrategyConfig(use_tiered_oracle=False, incremental_update=True)
