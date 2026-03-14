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

from ase import Atoms
from src.domain_models.config import SystemConfig, InterfaceTarget, PolicyConfig, StructureGeneratorConfig
from src.domain_models.dtos import MaterialFeatures
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine, FeatureExtractor
from src.generators.structure_generator import StructureGenerator

def test_uat_02_01_adaptive_exploration_policy_evaluation():
    """UAT-02-01: Verify Adaptive Exploration Policy Engine."""
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)

    # Metal Scenario
    features_metal = MaterialFeatures(elements=["Fe", "Pt"], melting_point=1000.0, band_gap=0.0)
    strategy_metal = engine.decide_policy(features_metal)
    assert strategy_metal.md_mc_ratio > 0
    assert strategy_metal.policy_name == "High-MC Policy"

    # Insulator Scenario
    features_insulator = MaterialFeatures(elements=["Mg", "O"], melting_point=1000.0, band_gap=2.0)
    strategy_insulator = engine.decide_policy(features_insulator)
    assert strategy_insulator.n_defects > 0
    assert strategy_insulator.policy_name == "Defect-Driven Policy"

def test_uat_02_02_robust_interface_generation():
    """UAT-02-02: Verify Robust Interface Generation."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    # Valid FePt/MgO
    target_valid = InterfaceTarget(element1="FePt", element2="MgO")
    interface_atoms = generator.generate_interface(target_valid)

    symbols = interface_atoms.get_chemical_symbols()
    assert "Fe" in symbols
    assert "Pt" in symbols
    assert "Mg" in symbols
    assert "O" in symbols

    # Invalid Unobtainium
    target_invalid = InterfaceTarget(element1="Unobtainium", element2="Fe")
    import pytest
    with pytest.raises(ValueError, match="Invalid or unsupported element target"):
        generator.generate_interface(target_invalid)

def test_uat_02_03_oom_protection():
    """UAT-02-03: Verify OOM Protection During Candidate Generation."""
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    # Massive object
    massive_atoms = Atoms("Fe" * 50000)
    import pytest
    with pytest.raises(ValueError, match="too large for rattling"):
        generator.generate_local_candidates(massive_atoms)

    # Moderately large object scaling
    mod_atoms = Atoms("Fe" * 2000, positions=[(0, 0, 0)] * 2000)
    mod_atoms.set_cell([10, 10, 10])
    candidates = generator.generate_local_candidates(mod_atoms, n=200)

    # max(1, min(n, 100) // 10) = 10
    assert len(candidates) == 10

def test_uat_02_04_autonomous_feature_extraction():
    """UAT-02-04: Verify Autonomous Feature Extraction (Cold Start)."""
    extractor = FeatureExtractor()
    features = extractor.extract_features(["Fe", "Pt"])

    assert features.band_gap == 0.0
    assert features.melting_point > 0.0

    # This proves the fallback works seamlessly without crashing when heavy MLIPs aren't present.
