from pathlib import Path

import pytest
from ase import Atoms

from src.domain_models.config import TrainerConfig
from src.trainers.ace_trainer import PacemakerWrapper


def test_pacemaker_initialization() -> None:
    config = TrainerConfig(max_epochs=10, active_set_size=200)
    wrapper = PacemakerWrapper(config)
    assert wrapper.config.max_epochs == 10


def test_update_dataset(tmp_path: Path) -> None:
    config = TrainerConfig()
    wrapper = PacemakerWrapper(config)

    atoms1 = Atoms("Fe", positions=[(0, 0, 0)])
    atoms2 = Atoms("Pt", positions=[(0, 0, 0)])

    dataset_path = tmp_path / "accumulated.extxyz"
    dataset_path = wrapper.update_dataset([atoms1, atoms2], dataset_path)
    assert dataset_path.exists()
    assert str(dataset_path).endswith(".extxyz")


def test_select_local_active_set(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TrainerConfig()
    wrapper = PacemakerWrapper(config)

    anchor = Atoms("Fe", positions=[(0, 0, 0)])
    candidates = [Atoms("Fe", positions=[(i * 0.1, 0, 0)]) for i in range(1, 21)]

    import subprocess
    from collections.abc import Sequence
    from typing import Any

    def mock_run(cmd: Sequence[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        # Mocking pace_activeset to write a dummy file.
        # In the builder, out_file is the 4th item because of quotes: ['pace_activeset', '--input', "'input_path'", '--output', "'output_path'", '--n', "'5'"]
        # To be robust, search for '--output'
        out_idx = cmd.index("--output") + 1
        out_file = Path(cmd[out_idx].strip("'\""))
        from ase.io import write

        # Write dummy
        write(str(out_file), [anchor, *candidates[:4]], format="extxyz")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    selected = wrapper.select_local_active_set(candidates, anchor=anchor, n=5)

    # Must include anchor
    assert len(selected) == 5

    assert isinstance(selected[0], Atoms)
