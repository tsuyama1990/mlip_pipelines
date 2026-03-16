import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from src.domain_models.config import TrainerConfig
from src.trainers.finetune_manager import FinetuneManager


@pytest.fixture
def mock_config():
    return TrainerConfig(trusted_directories=[])


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
        with pytest.raises(RuntimeError, match="mace_run_train completed but failed to produce a .model file."):
            finetune_manager.finetune_mace(structures, "base.model", output_path)


@patch("shutil.rmtree")
def test_finetune_mace_permission_error_cleanup(mock_rmtree, finetune_manager, tmp_path, monkeypatch):
    structures = [Atoms("Fe", positions=[(0, 0, 0)])]
    output_path = tmp_path / "output_model"

    def mock_run(cmd, *args, **kwargs):
        # We must create a mock output model file in the temp dir so the manager finds it
        out_dir_idx = cmd.index("--output_dir") + 1
        temp_dir = Path(cmd[out_dir_idx])
        (temp_dir / "mock.model").write_text("model weights")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    mock_rmtree.side_effect = PermissionError("Cannot delete dir")

    with patch.object(finetune_manager, "_resolve_binary_path", return_value="mace_run_train"):
        import unittest
        with patch("logging.warning") as mock_warning:
            result_path = finetune_manager.finetune_mace(structures, "base.model", output_path)

            # The function should succeed, log the warning, and not crash
            assert result_path.exists()
            mock_warning.assert_called_with(unittest.mock.ANY)
