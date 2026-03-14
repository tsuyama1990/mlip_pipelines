import subprocess
import sys
from pathlib import Path

import pytest


def test_uat_01_01_valid_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.orchestrator import Orchestrator
    from src.domain_models.config import ProjectConfig

    (tmp_path / "README.md").touch()

    env_content = f"""MLIP_SYSTEM__ELEMENTS='["Fe", "C"]'
MLIP_PROJECT_ROOT={tmp_path.resolve()}
MLIP_DYNAMICS__TRUSTED_DIRECTORIES=[]
MLIP_TRAINER__TRUSTED_DIRECTORIES=[]
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    env_file.chmod(0o600)

    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLIP_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("MLIP_SYSTEM__ELEMENTS", raising=False)

    # Needs to patch pyacemaker because we might not have it installed
    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )

    from src.domain_models.config import (
        DynamicsConfig,
        OracleConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )
    config = ProjectConfig(
        project_root=tmp_path,
        system=SystemConfig(elements=["Fe", "C"]),
        dynamics=DynamicsConfig(project_root=str(tmp_path), trusted_directories=[]),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig()
    )
    assert config.system.elements == ["Fe", "C"]

    orch = Orchestrator(config)
    # Mocking components
    from src.dynamics.dynamics_engine import MDInterface

    class MockMD(MDInterface):
        def run_exploration(self, *args: type, **kwargs: type) -> dict: # type: ignore
            work_dir = kwargs.get("work_dir")
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True) # type: ignore
                dump_file = work_dir / "dummy_dump" # type: ignore
                dump_file.touch() # type: ignore
                return {"halted": True, "dump_file": str(dump_file)}
            return {"halted": True, "dump_file": "dummy_dump"}

        def extract_high_gamma_structures(self, *args: type, **kwargs: type) -> list: # type: ignore
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])]

        def resume(self, potential: type, restart_dir: type, work_dir: type) -> dict: # type: ignore
            return {"halted": False, "dump_file": None}

    class MockOracle:
        def compute_batch(self, *args: type, **kwargs: type) -> list: # type: ignore
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])]

    class MockTrainer:
        def get_latest_potential(self) -> str:
            return "dummy_pot.yace"

        def select_local_active_set(self, *args: type, **kwargs: type) -> list: # type: ignore
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])] * 5

        def update_dataset(self, *args: type, **kwargs: type) -> Path: # type: ignore
            from pathlib import Path
            return Path("dummy.pckl") # type: ignore

        def train(self, dataset: type, initial_potential: type, output_dir: type) -> Path: # type: ignore
            pot = output_dir / "output_potential.yace" # type: ignore
            pot.parent.mkdir(parents=True, exist_ok=True)
            pot.write_text("elements version b_functions dummy potential")
            return pot # type: ignore

    class MockValidator:
        def validate(self, *args: type, **kwargs: type): # type: ignore
            from src.domain_models.dtos import ValidationReport
            return ValidationReport(
                passed=True,
                energy_rmse=0.001,
                force_rmse=0.01,
                stress_rmse=0.05,
                phonon_stable=True,
                mechanically_stable=True,
            )

    class MockGenerator:
        def generate_local_candidates(self, s0: type, n: int = 20) -> list: # type: ignore
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])] * n

    class MockPolicyEngine:
        def decide_policy(self, *args: type, **kwargs: type): # type: ignore
            from src.domain_models.dtos import ExplorationStrategy
            return ExplorationStrategy(
                md_mc_ratio=0.0,
                t_max=300.0,
                n_defects=0.0,
                strain_range=0.0,
                policy_name="Standard",
            )

    orch.md_engine = MockMD(config.dynamics, config.system)
    orch.oracle = MockOracle()  # type: ignore
    orch.trainer = MockTrainer()  # type: ignore
    orch.validator = MockValidator()  # type: ignore
    orch.structure_generator = MockGenerator()  # type: ignore
    orch.policy_engine = MockPolicyEngine()  # type: ignore

    res = orch.run_cycle()
    assert res is not None
    assert str(res).endswith("generation_001.yace")
    assert not list(tmp_path.glob("active_learning/tmp*"))  # Temp dirs should be cleaned up


def test_uat_01_02_invalid_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from src.domain_models.config import ProjectConfig

    (tmp_path / "README.md").touch()

    env_content = f"""MLIP_SYSTEM__ELEMENTS='["Fe", "C"]'
MLIP_PROJECT_ROOT={tmp_path.resolve()}/../
MLIP_DYNAMICS__TRUSTED_DIRECTORIES=[]
MLIP_TRAINER__TRUSTED_DIRECTORIES=[]
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    env_file.chmod(0o600)

    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLIP_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("MLIP_SYSTEM__ELEMENTS", raising=False)

    from src.domain_models.config import (
        DynamicsConfig,
        OracleConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )
    with pytest.raises(ValidationError, match=".*(Path traversal sequences|Invalid characters or traversal sequences).*"):
        ProjectConfig()


def test_uat_01_04_checkpointing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.orchestrator import Orchestrator
    from src.domain_models.config import ProjectConfig

    # Pre-populate directory
    pot_dir = tmp_path / "potentials"
    pot_dir.mkdir()
    (pot_dir / "generation_001.yace").touch()
    (pot_dir / "generation_002.yace").touch()
    (pot_dir / "generation_003.yace").touch()

    (tmp_path / "README.md").touch()

    env_content = f"""MLIP_SYSTEM__ELEMENTS='["Fe", "C"]'
MLIP_PROJECT_ROOT={tmp_path.resolve()}
MLIP_DYNAMICS__TRUSTED_DIRECTORIES=[]
MLIP_TRAINER__TRUSTED_DIRECTORIES=[]
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    env_file.chmod(0o600)

    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLIP_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("MLIP_SYSTEM__ELEMENTS", raising=False)

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )

    from src.domain_models.config import (
        DynamicsConfig,
        OracleConfig,
        SystemConfig,
        TrainerConfig,
        ValidatorConfig,
    )
    config = ProjectConfig(
        project_root=tmp_path,
        system=SystemConfig(elements=["Fe", "C"]),
        dynamics=DynamicsConfig(project_root=str(tmp_path), trusted_directories=[]),
        oracle=OracleConfig(),
        trainer=TrainerConfig(trusted_directories=[]),
        validator=ValidatorConfig()
    )
    orch = Orchestrator(config)

    assert orch.iteration == 3

    from src.dynamics.dynamics_engine import MDInterface
    class MockMD(MDInterface):
        def run_exploration(self, *args: type, **kwargs: type) -> dict: # type: ignore
            work_dir = kwargs.get("work_dir")
            if work_dir:
                work_dir.mkdir(parents=True, exist_ok=True) # type: ignore
                dump_file = work_dir / "dummy_dump" # type: ignore
                dump_file.touch() # type: ignore
                return {"halted": True, "dump_file": str(dump_file)}
            return {"halted": True, "dump_file": "dummy_dump"}

        def extract_high_gamma_structures(self, *args: type, **kwargs: type) -> list: # type: ignore
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])]

        def resume(self, potential: type, restart_dir: type, work_dir: type) -> dict: # type: ignore
            return {"halted": False, "dump_file": None}

    class MockOracle:
        def compute_batch(self, *args: type, **kwargs: type) -> list: # type: ignore
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])]

    class MockTrainer:
        def get_latest_potential(self) -> str:
            return "dummy_pot.yace"

        def select_local_active_set(self, *args: type, **kwargs: type) -> list: # type: ignore
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])] * 5

        def update_dataset(self, *args: type, **kwargs: type) -> Path: # type: ignore
            from pathlib import Path
            return Path("dummy.pckl") # type: ignore

        def train(self, dataset: type, initial_potential: type, output_dir: type) -> Path: # type: ignore
            pot = output_dir / "output_potential.yace" # type: ignore
            pot.parent.mkdir(parents=True, exist_ok=True)
            pot.write_text("elements version b_functions dummy potential")
            return pot # type: ignore

    class MockValidator:
        def validate(self, *args: type, **kwargs: type): # type: ignore
            from src.domain_models.dtos import ValidationReport
            return ValidationReport(
                passed=True,
                energy_rmse=0.001,
                force_rmse=0.01,
                stress_rmse=0.05,
                phonon_stable=True,
                mechanically_stable=True,
            )

    class MockGenerator:
        def generate_local_candidates(self, s0: type, n: int = 20) -> list: # type: ignore
            from ase import Atoms
            return [Atoms("Fe", positions=[(0, 0, 0)])] * n

    class MockPolicyEngine:
        def decide_policy(self, *args: type, **kwargs: type): # type: ignore
            from src.domain_models.dtos import ExplorationStrategy
            return ExplorationStrategy(
                md_mc_ratio=0.0,
                t_max=300.0,
                n_defects=0.0,
                strain_range=0.0,
                policy_name="Standard",
            )

    orch.md_engine = MockMD(config.dynamics, config.system)
    orch.oracle = MockOracle()  # type: ignore
    orch.trainer = MockTrainer()  # type: ignore
    orch.validator = MockValidator()  # type: ignore
    orch.structure_generator = MockGenerator()  # type: ignore
    orch.policy_engine = MockPolicyEngine()  # type: ignore

    res = orch.run_cycle()
    assert orch.iteration == 4
    assert str(res).endswith("generation_004.yace")


@pytest.mark.skip(
    reason="Headless execution of marimo notebooks can cause timeouts or missing dependency errors in CI"
)
def test_marimo_tutorial(tmp_path: Path) -> None:
    # Tests that the tutorial runs headlessly without errors
    tutorial_path = Path("tutorials/uat_and_tutorial.py")
    assert tutorial_path.exists()

    # Run the script directly since it's a valid python script via marimo
    res = subprocess.run(
        [sys.executable, "-m", "marimo", "run", str(tutorial_path)],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        check=False,
    )
    assert res.returncode == 0
