from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from src.domain_models.config import TrainerConfig
from src.trainers.ace_trainer import PacemakerWrapper


def test_pacemaker_initialization() -> None:
    config = TrainerConfig(max_epochs=10, active_set_size=200, trusted_directories=[])
    wrapper = PacemakerWrapper(config)
    assert wrapper.config.max_epochs == 10


def test_update_dataset(tmp_path: Path) -> None:
    config = TrainerConfig(trusted_directories=[])
    wrapper = PacemakerWrapper(config)

    atoms1 = Atoms("Fe", positions=[(0, 0, 0)])
    atoms2 = Atoms("Pt", positions=[(0, 0, 0)])

    dataset_path = tmp_path / "accumulated.extxyz"
    dataset_path = wrapper.update_dataset([atoms1, atoms2], dataset_path)
    assert dataset_path.exists()
    assert str(dataset_path).endswith(".extxyz")


def test_select_local_active_set(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TrainerConfig(trusted_directories=[])
    wrapper = PacemakerWrapper(config)

    anchor = Atoms("Fe", positions=[(0, 0, 0)])
    candidates = [Atoms("Fe", positions=[(i * 0.1, 0, 0)]) for i in range(1, 21)]

    import subprocess
    from collections.abc import Sequence
    from typing import Any

    class MockProcess:
        def __init__(self, args: Any, returncode: int, stdout: str, stderr: str) -> None:
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def mock_run(cmd: Sequence[str], *args: Any, **kwargs: Any) -> MockProcess:
        # Mocking pace_activeset to properly simulate subprocess behavior and write dummy files
        out_idx = cmd.index("--output") + 1
        out_file = Path(cmd[out_idx].strip("'\""))

        # Simulate pace_activeset input validation
        in_idx = cmd.index("--input") + 1
        in_file = Path(cmd[in_idx].strip("'\""))
        if not in_file.exists():
            return MockProcess(args=cmd, returncode=1, stdout="", stderr="Input file not found")

        from ase.io import write

        write(str(out_file), [anchor, *candidates[:4]], format="extxyz")

        return MockProcess(args=cmd, returncode=0, stdout="Success", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    selected = wrapper.select_local_active_set(candidates, anchor=anchor, n=5)

    # Must include anchor
    assert len(selected) == 5

    assert isinstance(selected[0], Atoms)


class TestACETrainer:
    @pytest.fixture
    def mock_config(self):
        from src.domain_models.config import TrainerConfig

        return TrainerConfig(trusted_directories=[])

    @pytest.fixture
    def ace_trainer(self, mock_config):
        from src.trainers.ace_trainer import PacemakerWrapper

        return PacemakerWrapper(mock_config)

    @pytest.fixture
    def anchor_atoms(self):
        from ase import Atoms

        return Atoms("Fe", positions=[(0, 0, 0)])

    @patch("subprocess.run")
    def test_select_local_active_set_failure_2(self, mock_run, ace_trainer, tmp_path, anchor_atoms):
        import subprocess

        # Test pace_activeset subprocess failure
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="Failed")
        candidates = [anchor_atoms.copy()]

        with pytest.raises(RuntimeError, match="pace_activeset failed"):
            ace_trainer.select_local_active_set(candidates, anchor_atoms, n=1)

    @patch("subprocess.run")
    def test_select_local_active_set_not_found_2(self, mock_run, ace_trainer, anchor_atoms):
        # Test executable missing
        mock_run.side_effect = FileNotFoundError("Executable not found")
        candidates = [anchor_atoms.copy()]

        with pytest.raises(RuntimeError, match="pace_activeset executable not found in PATH"):
            ace_trainer.select_local_active_set(candidates, anchor_atoms, n=1)

    @patch("subprocess.run")
    def test_select_local_active_set_no_output_2(
        self, mock_run, ace_trainer, tmp_path, anchor_atoms
    ):
        # Test it returns but doesn't create the file
        mock_run.return_value = MagicMock(returncode=0)
        candidates = [anchor_atoms.copy()]

        with pytest.raises(RuntimeError, match="pace_activeset did not generate the output file"):
            ace_trainer.select_local_active_set(candidates, anchor_atoms, n=1)

    def test_select_local_active_set_invalid_n_2(self, ace_trainer, anchor_atoms):
        candidates = [anchor_atoms.copy()]
        with pytest.raises(ValueError, match="n must be a positive integer"):
            ace_trainer.select_local_active_set(candidates, anchor_atoms, n=0)

    def test_train_dataset_not_found_2(self, ace_trainer, tmp_path):
        bad_dataset = tmp_path / "not_found.extxyz"
        out_dir = tmp_path / "out"
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            ace_trainer.train(bad_dataset, None, out_dir)

    def test_train_dataset_invalid_suffix_2(self, ace_trainer, tmp_path):
        bad_dataset = tmp_path / "dataset.txt"
        bad_dataset.write_text("10\ncontent")
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="Dataset must be an .extxyz file"):
            ace_trainer.train(bad_dataset, None, out_dir)

    def test_train_dataset_invalid_format_2(self, ace_trainer, tmp_path):
        bad_dataset = tmp_path / "dataset.extxyz"
        bad_dataset.write_text("not_a_number\ncontent")
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="Dataset does not appear to be a valid XYZ format"):
            ace_trainer.train(bad_dataset, None, out_dir)

    @patch("subprocess.run")
    def test_train_subprocess_failure_2(self, mock_run, ace_trainer, tmp_path):
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="Train Failed")

        dataset = tmp_path / "dataset.extxyz"
        dataset.write_text("10\ncontent")
        out_dir = tmp_path / "out"

        with pytest.raises(RuntimeError, match="pace_train execution failed"):
            ace_trainer.train(dataset, None, out_dir)

    @patch("subprocess.run")
    def test_train_subprocess_timeout(self, mock_run, ace_trainer, tmp_path):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 3600)

        dataset = tmp_path / "dataset.extxyz"
        dataset.write_text("10\ncontent")
        out_dir = tmp_path / "out"

        with pytest.raises(RuntimeError, match="pace_train execution timed out."):
            ace_trainer.train(dataset, None, out_dir)

    @patch("subprocess.run")
    def test_train_executable_not_found_2(self, mock_run, ace_trainer, tmp_path):
        mock_run.side_effect = FileNotFoundError("Not found")

        dataset = tmp_path / "dataset.extxyz"
        dataset.write_text("10\ncontent")
        out_dir = tmp_path / "out"

        with pytest.raises(RuntimeError, match="pace_train executable not found in PATH"):
            ace_trainer.train(dataset, None, out_dir)

    @patch("subprocess.run")
    def test_train_success_2(self, mock_run, ace_trainer, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        dataset = tmp_path / "dataset.extxyz"
        dataset.write_text("10\ncontent")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        init_pot = tmp_path / "init.yace"
        init_pot.write_text("pot")

        # Test full success path
        res_pot = ace_trainer.train(dataset, init_pot, out_dir)
        assert res_pot == out_dir / "output_potential.yace"

        # Validate that cmd passed to subprocess contains the initial potential argument
        cmd_args = mock_run.call_args[0][0]
        assert "--initial_potential" in cmd_args
        assert str(init_pot.resolve()) in cmd_args

    @patch("subprocess.run")
    def test_select_local_active_set_invalid_paths(
        self, mock_run, ace_trainer, tmp_path, anchor_atoms
    ):
        candidates = [anchor_atoms.copy()]
        # The trainer creates temp files internally, so we don't directly control the input/output paths it checks
        # But we can patch re.match to force invalid paths
        import re

        original_match = re.match

        def mock_match(pattern, string, flags=0):
            if pattern == r"^[/a-zA-Z0-9_.-]+$":
                return None  # Force failure
            return original_match(pattern, string, flags)

        with patch("re.match", side_effect=mock_match):
            with pytest.raises(ValueError, match="Invalid input file path"):
                ace_trainer.select_local_active_set(candidates, anchor_atoms, n=1)

    @patch("subprocess.run")
    def test_train_invalid_baseline(self, mock_run, ace_trainer, tmp_path):
        dataset = tmp_path / "dataset.extxyz"
        dataset.write_text("10\ncontent")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        ace_trainer.config.baseline_potential = "invalid baseline"
        with pytest.raises(ValueError, match="Invalid baseline potential format"):
            ace_trainer.train(dataset, None, out_dir)

    @patch("subprocess.run")
    def test_train_invalid_regularization(self, mock_run, ace_trainer, tmp_path):
        dataset = tmp_path / "dataset.extxyz"
        dataset.write_text("10\ncontent")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        ace_trainer.config.regularization = "invalid reg!"
        with pytest.raises(ValueError, match="Invalid regularization format"):
            ace_trainer.train(dataset, None, out_dir)

    @patch("subprocess.run")
    def test_train_success_incremental(self, mock_run, ace_trainer, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        dataset = tmp_path / "dataset.extxyz"
        dataset.write_text("10\ncontent")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        init_pot = tmp_path / "init.yace"
        init_pot.write_text("pot")

        ace_trainer.config.max_epochs = 100

        # Test full success path
        res_pot = ace_trainer.train(dataset, init_pot, out_dir)
        assert res_pot == out_dir / "output_potential.yace"

        # Validate that cmd passed to subprocess contains the initial potential argument
        cmd_args = mock_run.call_args[0][0]
        assert "--initial_potential" in cmd_args
        assert str(init_pot.resolve()) in cmd_args

        # Max epochs should be scaled down by 10
        assert "--max_num_epochs" in cmd_args
        epochs_idx = cmd_args.index("--max_num_epochs") + 1
        assert cmd_args[epochs_idx] == "10"

    def test_manage_replay_buffer(self, ace_trainer, tmp_path):
        from ase import Atoms
        from ase.io import write

        history_file = tmp_path / "history.extxyz"

        # Create history file with 10 atoms
        hist_atoms = [Atoms("Cu", positions=[(i, 0, 0)]) for i in range(10)]
        write(str(history_file), hist_atoms, format="extxyz")

        # New atoms
        new_atoms = [Atoms("Fe", positions=[(i, 0, 0)]) for i in range(3)]

        # Sample with buffer_size = 5
        combined = ace_trainer._manage_replay_buffer(new_atoms, history_file, buffer_size=5)

        # Combined should be 5 + 3 = 8
        assert len(combined) == 8

        # History file should now have 13 atoms
        from ase.io import read

        updated_history = read(str(history_file), index=":")
        assert len(updated_history) == 13

    def test_manage_replay_buffer_small_history(self, ace_trainer, tmp_path):
        from ase import Atoms
        from ase.io import write

        history_file = tmp_path / "history.extxyz"

        # Create history file with 2 atoms
        hist_atoms = [Atoms("Cu", positions=[(i, 0, 0)]) for i in range(2)]
        write(str(history_file), hist_atoms, format="extxyz")

        # New atoms
        new_atoms = [Atoms("Fe", positions=[(i, 0, 0)]) for i in range(3)]

        # Sample with buffer_size = 5
        combined = ace_trainer._manage_replay_buffer(new_atoms, history_file, buffer_size=5)

        # Combined should be 2 + 3 = 5
        assert len(combined) == 5
