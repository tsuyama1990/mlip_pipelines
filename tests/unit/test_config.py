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


def test_dynamics_config_valid() -> None:
    config = DynamicsConfig(
        uncertainty_threshold=10.0, project_root=str(Path.cwd()), trusted_directories=[]
    )
    assert config.uncertainty_threshold == 10.0


def test_oracle_config_invalid() -> None:
    with pytest.raises(ValidationError):
        OracleConfig(kspacing=-0.1)  # gt=0.0 constraint violated


def test_project_config(tmp_path: Path) -> None:
    # Should be valid with minimal required setup (since most have defaults)
    (tmp_path / "README.md").touch()

    config = ProjectConfig(
        project_root=tmp_path,
        system=SystemConfig(elements=["Fe", "O"]),
        dynamics=DynamicsConfig(project_root=str(Path.cwd()), trusted_directories=[]),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig(),
    )
    assert config.system.elements == ["Fe", "O"]
    assert config.system.baseline_potential == "zbl"


def test_validate_single_trusted_dir_valid(tmp_path: Path) -> None:
    d = tmp_path / "valid_dir"
    d.mkdir()
    from src.domain_models.config import _validate_single_trusted_dir

    assert _validate_single_trusted_dir(str(d)) == str(d.resolve())


def test_validate_single_trusted_dir_not_exist(tmp_path: Path) -> None:
    d = tmp_path / "not_exist"
    from src.domain_models.config import _validate_single_trusted_dir

    assert _validate_single_trusted_dir(str(d)) is None


def test_validate_single_trusted_dir_not_dir(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("dummy")
    from src.domain_models.config import _validate_single_trusted_dir

    with pytest.raises(ValueError, match="(?i).*must be a directory.*"):
        _validate_single_trusted_dir(str(f))


def test_validate_kspacing() -> None:
    from pydantic import ValidationError

    from src.domain_models.config import OracleConfig

    with pytest.raises(ValidationError, match=".*kspacing.*"):
        OracleConfig(kspacing=0.0)


def test_project_config_env_key() -> None:
    from src.domain_models.config import ProjectConfig

    with pytest.raises(ValueError, match=".*Unauthorized environment variable injected via .env.*"):
        ProjectConfig._validate_env_key("INVALID KEY")


def test_project_config_env_value() -> None:
    from src.domain_models.config import ProjectConfig

    with pytest.raises(ValueError, match=".*Invalid characters or traversal sequences.*"):
        ProjectConfig._validate_env_value("value; rm -rf")
    with pytest.raises(ValueError, match=".*Invalid characters or traversal sequences.*"):
        ProjectConfig._validate_env_value("../secret")


def test_project_config_env_file_security(tmp_path: Path) -> None:
    from src.domain_models.config import ProjectConfig

    base = tmp_path / "base"
    base.mkdir()
    env = base / ".env"

    with pytest.raises(FileNotFoundError, match=".*"):
        ProjectConfig._validate_env_file_security(env, base)

    env.mkdir()
    with pytest.raises(ValueError, match=".*"):
        ProjectConfig._validate_env_file_security(env, base)

    env.rmdir()
    target = base / "target.txt"
    target.write_text("test")
    from pathlib import Path

    Path(env).symlink_to(target)
    with pytest.raises(ValueError, match=".*must not be a symlink.*"):
        ProjectConfig._validate_env_file_security(env, base)


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
