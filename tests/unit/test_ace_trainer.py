import pytest
from pathlib import Path
from ase.build import bulk
from unittest.mock import patch, MagicMock
from src.domain_models.config import TrainingConfig
from src.trainers.ace_trainer import ACETrainer


@patch("shutil.which", return_value="pace_activeset")
@patch("subprocess.run")
def test_ace_trainer_select_local_active_set(mock_run: MagicMock, mock_which: MagicMock) -> None:
    config = TrainingConfig()
    trainer = ACETrainer(config)

    anchor = bulk("Fe", cubic=True)
    candidates = [bulk("Pt", cubic=True) for _ in range(10)]

    selected = trainer.select_local_active_set(candidates, anchor, n=5)

    assert len(selected) == 5
    assert selected[0] is anchor
    mock_run.assert_called_once()


@patch("shutil.which", return_value=None)
def test_ace_trainer_select_local_active_set_no_pace(mock_which: MagicMock) -> None:
    config = TrainingConfig()
    trainer = ACETrainer(config)

    anchor = bulk("Fe", cubic=True)
    candidates = [bulk("Pt", cubic=True) for _ in range(10)]

    selected = trainer.select_local_active_set(candidates, anchor, n=5)

    assert len(selected) == 5
    assert selected[0] is anchor


def test_ace_trainer_select_local_active_set_few_candidates() -> None:
    config = TrainingConfig()
    trainer = ACETrainer(config)

    anchor = bulk("Fe", cubic=True)
    candidates = [bulk("Pt", cubic=True) for _ in range(2)]

    selected = trainer.select_local_active_set(candidates, anchor, n=5)

    assert len(selected) == 3
    assert selected[0] is anchor
    assert selected[1] is candidates[0] or selected[1] is candidates[1]


def test_ace_trainer_update_dataset(tmp_path: Path) -> None:
    config = TrainingConfig()
    trainer = ACETrainer(config)

    data = [bulk("Fe", cubic=True)]

    with patch("src.trainers.ace_trainer.Path") as mock_path_cls:
        # We need Path("data") to return a mock path that behaves like a path.
        # But patching Path directly is tricky because it returns tmp_path entirely,
        # so any Path("...") call gets tmp_path.
        # Let's just create a new directory inside tmp_path for the real method to use if we patch cwd or similar.
        pass

    import os
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        dataset_path = trainer.update_dataset(data)
        assert dataset_path.name == "accumulated.extxyz"
        assert dataset_path.parent.name == "data"
    finally:
        os.chdir(orig_cwd)


@patch("shutil.which", return_value="pace_train")
@patch("subprocess.run")
def test_ace_trainer_train(mock_run: MagicMock, mock_which: MagicMock, tmp_path: Path) -> None:
    config = TrainingConfig()
    trainer = ACETrainer(config)

    output_dir = tmp_path / "training_output"

    # Test without initial potential
    pot_path = trainer.train(Path("dummy_dataset.extxyz"), None, output_dir)
    assert pot_path.name == "output_potential.yace"
    mock_run.assert_called_once()

    # Test with initial potential
    initial_pot = tmp_path / "initial.yace"
    initial_pot.touch()
    pot_path = trainer.train(Path("dummy_dataset.extxyz"), initial_pot, output_dir)
    assert pot_path.name == "output_potential.yace"
    assert mock_run.call_count == 2
