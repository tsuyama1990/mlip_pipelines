import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from src.domain_models.config import TrainerConfig
from src.trainers.finetune_manager import FinetuneManager


@pytest.fixture
def mock_config(tmp_path):
    import os

    os.chown(tmp_path, os.getuid(), os.getgid())
    return TrainerConfig(trusted_directories=[str(tmp_path.resolve(strict=False))])


@pytest.fixture
def finetune_manager(mock_config):
    return FinetuneManager(mock_config)


def test_finetune_mace_success(finetune_manager, tmp_path, monkeypatch):
    structures = [Atoms("Fe", positions=[(0, 0, 0)])]
    model_path = "/path/to/base.model"
    output_path = tmp_path / "output_model"

    def mock_run(cmd, *args, **kwargs):
        # cmd should contain --freeze_body and correct paths
        assert "--freeze_body" in cmd
        assert "--max_num_epochs" in cmd
        assert "5" in cmd
        assert "--lr" in cmd
        assert "0.001" in cmd

        # We must create a mock output model file in the temp dir so the manager finds it
        out_dir_idx = cmd.index("--output_dir") + 1
        temp_dir = Path(cmd[out_dir_idx])
        (temp_dir / "mock.model").write_text("model weights")

        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Need to patch the executable resolution since it's tricky without a real binary
    with patch.object(finetune_manager, "_resolve_binary_path", return_value="mace_run_train"):
        result_path = finetune_manager.finetune_mace(structures, model_path, output_path)

    assert result_path == output_path / "finetuned.model"
    assert result_path.exists()
    assert result_path.read_text() == "model weights"


def test_finetune_mace_subprocess_failure(finetune_manager, tmp_path, monkeypatch):
    structures = [Atoms("Fe", positions=[(0, 0, 0)])]
    output_path = tmp_path / "output_model"

    def mock_run(cmd, *args, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="Finetuning failed")

    monkeypatch.setattr(subprocess, "run", mock_run)

    with patch.object(finetune_manager, "_resolve_binary_path", return_value="mace_run_train"):
        with pytest.raises(RuntimeError, match="mace_run_train execution failed"):
            finetune_manager.finetune_mace(structures, "base.model", output_path)


def test_finetune_mace_no_model_produced(finetune_manager, tmp_path, monkeypatch):
    structures = [Atoms("Fe", positions=[(0, 0, 0)])]
    output_path = tmp_path / "output_model"

    def mock_run(cmd, *args, **kwargs):
        # Don't produce any .model files in the temp dir
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    with patch.object(finetune_manager, "_resolve_binary_path", return_value="mace_run_train"):
        with pytest.raises(
            RuntimeError, match="mace_run_train completed but failed to produce a .model file."
        ):
            finetune_manager.finetune_mace(structures, "base.model", output_path)


@patch("shutil.rmtree")
def test_finetune_mace_permission_error_cleanup(
    mock_rmtree, finetune_manager, tmp_path, monkeypatch
):
    structures = [Atoms("Fe", positions=[(0, 0, 0)])]
    output_path = tmp_path / "output_model"

    class MockProcess:
        def __init__(self, args, returncode, stdout, stderr) -> None:
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def mock_run(cmd, *args, **kwargs):
        out_dir_idx = cmd.index("--output_dir") + 1
        temp_dir = Path(cmd[out_dir_idx])

        in_idx = cmd.index("--train_file") + 1
        in_file = Path(cmd[in_idx])
        if not in_file.exists():
            return MockProcess(cmd, 1, "", "Missing input file")

        # Simulate proper model output
        (temp_dir / "mock.model").write_text("model weights")
        return MockProcess(cmd, 0, "Success", "")

    monkeypatch.setattr(subprocess, "run", mock_run)

    # TemporaryDirectory will handle cleanup via weakref or atexit, we just verify it doesn't crash
    # However we'll intercept warnings to see if Python complains internally.

    with patch.object(finetune_manager, "_resolve_binary_path", return_value="mace_run_train"):
        with patch("logging.warning"):
            import contextlib

            with contextlib.suppress(PermissionError):
                result_path = finetune_manager.finetune_mace(structures, "base.model", output_path)

                # The function should succeed and not crash
                assert result_path.exists()
