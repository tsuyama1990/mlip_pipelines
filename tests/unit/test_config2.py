import tempfile


def test_config_misc():
    import pytest

    from src.domain_models.config import CutoutConfig

    with pytest.raises(Exception, match=".*"):
        CutoutConfig(core_radius=0.0)
    with pytest.raises(Exception, match=".*"):
        CutoutConfig(buffer_radius=0.0)
    with pytest.raises(Exception, match=".*"):
        CutoutConfig(core_radius=5.0, buffer_radius=4.0)


def test_config_misc2():
    import pytest

    from src.domain_models.config import DistillationConfig

    with pytest.raises(Exception, match=".*"):
        DistillationConfig(
            mace_model_path="../model.pt",
            temp_dir=tempfile.mkdtemp(),
            output_dir=tempfile.mkdtemp(),
            model_storage_path=tempfile.mkdtemp(),
        )
    with pytest.raises(Exception, match=".*"):
        DistillationConfig(
            mace_model_path="unknown_model",
            temp_dir=tempfile.mkdtemp(),
            output_dir=tempfile.mkdtemp(),
            model_storage_path=tempfile.mkdtemp(),
        )
    with pytest.raises(Exception, match=".*"):
        DistillationConfig(
            uncertainty_threshold=-0.1,
            temp_dir=tempfile.mkdtemp(),
            output_dir=tempfile.mkdtemp(),
            model_storage_path=tempfile.mkdtemp(),
        )
    with pytest.raises(Exception, match=".*"):
        DistillationConfig(
            sampling_structures_per_system=0,
            temp_dir=tempfile.mkdtemp(),
            output_dir=tempfile.mkdtemp(),
            model_storage_path=tempfile.mkdtemp(),
        )


def test_config_misc3():
    import pytest

    from src.domain_models.config import LoopStrategyConfig

    with pytest.raises(Exception, match=".*"):
        LoopStrategyConfig(
            incremental_update=True,
            use_tiered_oracle=False,
            replay_buffer_size=10,
            checkpoint_interval=1,
            timeout_seconds=10,
        )


def test_config_misc4(tmp_path):
    import pytest

    from src.domain_models.config import (
        ProjectConfig,
    )

    with pytest.raises(Exception, match=".*"):
        ProjectConfig(project_root=str(tmp_path / ".."))


def test_config_misc5():
    import pytest

    from src.domain_models.config import _validate_env_key

    with pytest.raises(Exception, match=".*"):
        _validate_env_key("INVALID")
    with pytest.raises(Exception, match=".*"):
        _validate_env_key("MLIP_" + "A" * 100)
    with pytest.raises(Exception, match=".*"):
        _validate_env_key("MLIP_A$")
    with pytest.raises(Exception, match=".*"):
        _validate_env_key("MLIP_a")


def test_config_misc6():
    import pytest

    from src.domain_models.config import _validate_env_value

    with pytest.raises(Exception, match=".*"):
        _validate_env_value("A" * 300)
    with pytest.raises(Exception, match=".*"):
        _validate_env_value("A/../B")
    with pytest.raises(Exception, match=".*"):
        _validate_env_value("A;B")
    with pytest.raises(Exception, match=".*"):
        _validate_env_value("A B")


def test_config_misc7(monkeypatch, tmp_path):
    import pytest

    from src.domain_models.config import _validate_env_permissions_and_size

    with pytest.raises(Exception, match=".*"):
        _validate_env_permissions_and_size(tmp_path / "non_existent.env")
    (tmp_path / "env.link").symlink_to("foo")
    with pytest.raises(Exception, match=".*"):
        _validate_env_permissions_and_size(tmp_path / "env.link")
    (tmp_path / "valid.env").touch()
    (tmp_path / "valid.env").chmod(0o777)
    with pytest.raises(Exception, match=".*"):
        _validate_env_permissions_and_size(tmp_path / "valid.env")
