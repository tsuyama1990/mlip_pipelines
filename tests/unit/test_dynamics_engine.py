from pathlib import Path
from typing import Any

import pytest

from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.dynamics_engine import MDInterface


def test_md_interface_initialization(tmp_path: Path) -> None:
    config = DynamicsConfig(uncertainty_threshold=6.0, md_steps=1000, temperature=300.0)
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config, project_root=tmp_path)
    assert engine.config.uncertainty_threshold == 6.0


def test_md_run_exploration_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config, project_root=tmp_path)

    # Mock LAMMPS run
    def mock_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"halted": True, "dump_file": tmp_path / "dump.lammps"}

    monkeypatch.setattr(engine, "run_exploration", mock_run)

    result = engine.run_exploration(potential=Path("dummy.yace"), work_dir=tmp_path)
    assert result["halted"] is True
    assert result["dump_file"] == tmp_path / "dump.lammps"


def test_extract_high_gamma_structures(tmp_path: Path) -> None:
    config = DynamicsConfig()
    sys_config = SystemConfig(elements=["Fe", "Pt"])
    engine = MDInterface(config, sys_config, project_root=tmp_path)

    dump_file = tmp_path / "dump.lammps"
    # Basic LAMMPS dump content with 1 atom
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
"""
    dump_file.write_text(dump_content)

    structures = engine.extract_high_gamma_structures(dump_file, 5.0)
    assert isinstance(structures, list)
    assert len(structures) == 1
    assert len(structures[0]) == 1
