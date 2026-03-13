import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.domain_models.config import MaterialConfig, MDConfig, OTFLoopConfig
from src.domain_models.dtos import ExplorationStrategy
from src.dynamics.dynamics_engine import DynamicsEngine


@patch.dict(sys.modules, {"lammps": MagicMock()})
def test_dynamics_engine_run_exploration_lammps(mock_material_config: MaterialConfig, tmp_path: Path) -> None:
    # Set up lammps mock
    mock_lammps = sys.modules["lammps"]
    lmp_instance = MagicMock()
    mock_lammps.lammps.return_value = lmp_instance

    # We want max_gamma to be extracted as 6.0
    lmp_instance.extract_variable.return_value = 6.0

    md_config = MDConfig(steps=1000)
    otf_config = OTFLoopConfig(uncertainty_threshold=5.0)
    material = mock_material_config
    engine = DynamicsEngine(md_config, otf_config, material)
    strategy = ExplorationStrategy()

    result = engine.run_exploration(Path("dummy.yace"), strategy, tmp_path)

    assert (
        result["halted"] is False
    )  # Unless an exception is thrown inside command execution, it continues normally?
    # Wait, the fallback returned halted=True based on secrets check, but the LAMMPS loop
    # only sets halted=True if an exception occurs during the command execution.
    # So if max_gamma = 6.0, the script continues normally until Lammps throws "error hard",
    # which is handled by Lammps python bindings throwing an exception.
    # Let's verify the logic:

    # If we throw an exception from lmp.command...


@patch.dict(sys.modules, {"lammps": MagicMock()})
def test_dynamics_engine_run_exploration_lammps_halt(mock_material_config: MaterialConfig, tmp_path: Path) -> None:
    mock_lammps = sys.modules["lammps"]
    lmp_instance = MagicMock()
    mock_lammps.lammps.return_value = lmp_instance

    # Lammps throws exception on run when halt triggers error hard
    def mock_command(cmd: str) -> None:
        if "run" in cmd:
            msg = "LAMMPS error hard triggered by watchdog"
            raise RuntimeError(msg)

    lmp_instance.command.side_effect = mock_command
    lmp_instance.extract_variable.return_value = 6.0

    md_config = MDConfig(steps=1000)
    otf_config = OTFLoopConfig(uncertainty_threshold=5.0)
    material = mock_material_config
    engine = DynamicsEngine(md_config, otf_config, material)
    strategy = ExplorationStrategy()

    result = engine.run_exploration(Path("dummy.yace"), strategy, tmp_path)

    assert result["halted"] is True
    assert result["max_gamma"] == 6.0
    assert result["dump_file"] == tmp_path / "dump.lammps"


def test_dynamics_engine_run_exploration_fallback(mock_material_config: MaterialConfig, tmp_path: Path) -> None:
    # Ensure lammps import fails
    import sys

    with patch.dict(sys.modules, {"lammps": None}):
        md_config = MDConfig(steps=1000)
        otf_config = OTFLoopConfig(uncertainty_threshold=5.0)
        material = mock_material_config
        engine = DynamicsEngine(md_config, otf_config, material)
        strategy = ExplorationStrategy()

        with patch("secrets.SystemRandom.uniform", return_value=6.0):
            result = engine.run_exploration(Path("dummy.yace"), strategy, tmp_path)

        assert result["halted"] is True
        assert result["max_gamma"] == 6.0
        assert result["halt_step"] == 800
        assert result["dump_file"] == tmp_path / "dump.lammps"
        assert (tmp_path / "dump.lammps").parent.exists()


def test_dynamics_engine_extract_high_gamma_structures(mock_material_config: MaterialConfig, tmp_path: Path) -> None:
    md_config = MDConfig()
    otf_config = OTFLoopConfig()
    material = mock_material_config
    engine = DynamicsEngine(md_config, otf_config, material)

    dump_file = tmp_path / "dump.lammps"
    dump_file.write_text("dummy")

    atoms_list = engine.extract_high_gamma_structures(dump_file, threshold=5.0)

    assert len(atoms_list) == 1
    assert str(atoms_list[0].symbols) == "Fe2"
    assert len(atoms_list[0]) > 0
