from pathlib import Path
from typing import Any

import pytest

from src.core.orchestrator import Orchestrator
from src.domain_models.config import ProjectConfig
from src.domain_models.dtos import ValidationReport


def test_orchestrator_initialization(
    mock_project_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)
    assert orch.config.system.elements == ["Fe", "Pt"]
    assert orch.iteration == 0


def test_orchestrator_oracle_convergence_error(
    mock_project_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies OracleConvergenceError usage through the Orchestrator via dependency injection."""
    import sys

    from ase import Atoms

    from src.core import AbstractOracle
    from src.core.exceptions import OracleConvergenceError

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)

    class FailingOracle(AbstractOracle):
        def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
            msg = "Mocked SCF failure"
            raise OracleConvergenceError(msg)

    orch.oracle = FailingOracle()

    with pytest.raises(OracleConvergenceError, match="Mocked SCF failure"):
        # We manually call compute_batch to simulate the orchestrator hitting the error
        orch.oracle.compute_batch([], Path("dummy"))


def test_orchestrator_dynamics_halt_interrupt(
    mock_project_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies DynamicsHaltInterrupt usage through the Orchestrator via dependency injection."""
    import sys

    from src.core import AbstractDynamics
    from src.core.exceptions import DynamicsHaltInterrupt

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)

    class FailingDynamics(AbstractDynamics):
        def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:
            msg = "Mocked Halt"
            raise DynamicsHaltInterrupt(msg)

    orch.md_engine = FailingDynamics()

    with pytest.raises(DynamicsHaltInterrupt, match="Mocked Halt"):
        orch.md_engine.run_exploration(None, Path("dummy"))


def test_run_cycle(monkeypatch: pytest.MonkeyPatch, mock_project_config: ProjectConfig) -> None:  # noqa: C901
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)

    # Mock all internal models
    from src.dynamics.dynamics_engine import MDInterface

    class MockMD(MDInterface):
        def run_exploration(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            work_dir = kwargs.get("work_dir")
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True)
            return {"halted": True, "dump_file": "dummy_dump"}

        def extract_high_gamma_structures(self, *args: Any, **kwargs: Any) -> list[Any]:
            from ase import Atoms

            return [Atoms("Fe", positions=[(0, 0, 0)])]

        def resume(self, potential: Path, restart_dir: Path, work_dir: Path) -> dict[str, Any]:
            return {"halted": False, "dump_file": None}

    class MockOracle:
        def compute_batch(self, *args: Any, **kwargs: Any) -> list[Any]:
            from ase import Atoms

            return [Atoms("Fe", positions=[(0, 0, 0)])]

    class MockTrainer:
        def get_latest_potential(self) -> str:
            return "dummy_pot.yace"

        def select_local_active_set(self, *args: Any, **kwargs: Any) -> list[Any]:
            from ase import Atoms

            return [Atoms("Fe", positions=[(0, 0, 0)])] * 5

        def update_dataset(self, *args: Any, **kwargs: Any) -> Path:
            from pathlib import Path

            return Path("dummy.pckl")

        def train(self, dataset: Any, initial_potential: Any, output_dir: Path) -> Path:
            pot = output_dir / "output_potential.yace"
            pot.parent.mkdir(parents=True, exist_ok=True)
            pot.write_text("elements version dummy potential")
            return pot

    class MockValidator:
        def validate(self, *args: Any, **kwargs: Any) -> ValidationReport:
            from src.domain_models.dtos import ValidationReport

            return ValidationReport(
                passed=True,
                energy_rmse=0.001,
                force_rmse=0.01,
                stress_rmse=0.05,
                phonon_stable=True,
                mechanically_stable=True,
            )

    orch.md_engine = MockMD(mock_project_config.dynamics, mock_project_config.system)
    orch.oracle = MockOracle()  # type: ignore[assignment]
    orch.trainer = MockTrainer()  # type: ignore[assignment]
    orch.validator = MockValidator()  # type: ignore[assignment]

    # Mock structure generator
    class MockGenerator:
        def generate_local_candidates(self, s0: Any, n: int = 20) -> list[Any]:
            from ase import Atoms

            return [Atoms("Fe", positions=[(0, 0, 0)])] * n

    orch.structure_generator = MockGenerator()  # type: ignore[assignment]

    class MockPolicyEngine:
        def decide_policy(self, *args: Any, **kwargs: Any) -> Any:
            from src.domain_models.dtos import ExplorationStrategy

            return ExplorationStrategy(
                md_mc_ratio=0.0,  # Run MD
                t_max=300.0,
                n_defects=0.0,
                strain_range=0.0,
                policy_name="Standard",
            )

    orch.policy_engine = MockPolicyEngine()  # type: ignore[assignment]

    res = orch.run_cycle()
    assert orch.iteration == 1
    assert res is not None
    assert str(res).endswith("generation_001.yace")


def test_run_cycle_converged(
    monkeypatch: pytest.MonkeyPatch, mock_project_config: ProjectConfig
) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)

    from src.dynamics.dynamics_engine import MDInterface

    class MockMD(MDInterface):
        def run_exploration(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"halted": False, "dump_file": "dummy_dump"}

    class MockTrainer:
        def get_latest_potential(self) -> str:
            return "dummy_pot.yace"

    class MockPolicyEngine:
        def decide_policy(self, *args: Any, **kwargs: Any) -> Any:
            from src.domain_models.dtos import ExplorationStrategy

            return ExplorationStrategy(
                md_mc_ratio=0.0,  # Run MD
                t_max=300.0,
                n_defects=0.0,
                strain_range=0.0,
                policy_name="Standard",
            )

    orch.md_engine = MockMD(mock_project_config.dynamics, mock_project_config.system)
    orch.trainer = MockTrainer()  # type: ignore[assignment]
    orch.policy_engine = MockPolicyEngine()  # type: ignore[assignment]

    res = orch.run_cycle()
    assert res == "CONVERGED"


def test_get_latest_potential(
    monkeypatch: pytest.MonkeyPatch, mock_project_config: ProjectConfig, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)
    pot_dir = tmp_path / "potentials"
    pot_dir.mkdir(parents=True)
    pot_path = pot_dir / "generation_000.yace"
    pot_path.write_text("elements version")

    orch.config.project_root = tmp_path

    latest = orch.get_latest_potential()
    assert latest == pot_path.resolve()


def test_get_latest_potential_no_dir(
    monkeypatch: pytest.MonkeyPatch, mock_project_config: ProjectConfig, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)
    orch.config.project_root = tmp_path

    latest = orch.get_latest_potential()
    assert latest is None


def test_get_latest_potential_no_files(
    monkeypatch: pytest.MonkeyPatch, mock_project_config: ProjectConfig, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)
    pot_dir = tmp_path / "potentials"
    pot_dir.mkdir(parents=True)

    orch.config.project_root = tmp_path

    latest = orch.get_latest_potential()
    assert latest is None


def test_get_latest_potential_invalid_file(
    monkeypatch: pytest.MonkeyPatch, mock_project_config: ProjectConfig, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )
    orch = Orchestrator(mock_project_config)
    pot_dir = tmp_path / "potentials"
    pot_dir.mkdir(parents=True)
    pot_path = pot_dir / "generation_000.yace"
    pot_path.write_text("invalid data")

    orch.config.project_root = tmp_path

    latest = orch.get_latest_potential()
    assert latest is None
