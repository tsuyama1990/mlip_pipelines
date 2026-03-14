import pytest
from ase import Atoms
from pydantic import ValidationError

from src.domain_models.dtos import ExplorationStrategy, HaltInfo, MaterialFeatures, ValidationReport


def test_material_features_valid() -> None:
    feat = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    assert feat.elements == ["Fe", "Pt"]


def test_material_features_invalid() -> None:
    with pytest.raises(ValidationError):
        MaterialFeatures(elements=["Si"], band_gap=-1.0)  # ge=0.0

    with pytest.raises(ValidationError):
        MaterialFeatures(elements=["Fe"], extra_field="bad")  # type: ignore[call-arg]


def test_exploration_strategy() -> None:
    strategy = ExplorationStrategy(policy_name="Defect-Driven Policy", n_defects=0.05)
    assert strategy.policy_name == "Defect-Driven Policy"
    assert strategy.n_defects == 0.05


def test_halt_info_with_atoms() -> None:
    atoms1 = Atoms("Fe", positions=[(0, 0, 0)])
    atoms2 = Atoms("Pt", positions=[(0, 0, 0)])

    halt = HaltInfo(halted=True, high_gamma_atoms=[atoms1, atoms2], max_gamma=6.5)
    assert halt.halted is True
    assert halt.max_gamma == 6.5
    assert halt.high_gamma_atoms is not None
    assert len(halt.high_gamma_atoms) == 2


def test_validation_report_invalid() -> None:
    with pytest.raises(ValidationError):
        ValidationReport(
            passed=True,
            energy_rmse=0.005,
            force_rmse=0.03,
            stress_rmse=0.1,
            phonon_stable=True,
            mechanically_stable=True,
            extra_field="invalid",  # type: ignore[call-arg]
        )


def test_oracle_convergence_error() -> None:
    from src.core.exceptions import OracleConvergenceError

    msg = "SCF failed to converge after maximum retries"
    exc = OracleConvergenceError(msg)

    assert isinstance(exc, Exception)
    assert str(exc) == msg


def test_abstract_oracle_enforcement() -> None:
    from src.core import AbstractOracle

    class DummyOracle(AbstractOracle):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class DummyOracle"):
        DummyOracle()  # type: ignore[abstract]


def test_abstract_trainer_enforcement() -> None:
    from src.core import AbstractTrainer

    class DummyTrainer(AbstractTrainer):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class DummyTrainer"):
        DummyTrainer()  # type: ignore[abstract]


def test_abstract_dynamics_enforcement() -> None:
    from src.core import AbstractDynamics

    class DummyDynamics(AbstractDynamics):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class DummyDynamics"):
        DummyDynamics()  # type: ignore[abstract]


def test_abstract_generator_enforcement() -> None:
    from src.core import AbstractGenerator

    class DummyGenerator(AbstractGenerator):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class DummyGenerator"):
        DummyGenerator()  # type: ignore[abstract]


def test_concrete_interface_compliance() -> None:
    from pathlib import Path

    from src.core import AbstractDynamics, AbstractGenerator, AbstractOracle, AbstractTrainer
    from src.domain_models.config import (
        DynamicsConfig,
        OracleConfig,
        StructureGeneratorConfig,
        SystemConfig,
        TrainerConfig,
    )
    from src.dynamics.dynamics_engine import MDInterface
    from src.dynamics.eon_wrapper import EONWrapper
    from src.generators.structure_generator import StructureGenerator
    from src.oracles.dft_oracle import DFTManager
    from src.trainers.ace_trainer import PacemakerWrapper

    sys_cfg = SystemConfig(elements=["Fe", "Pt"], baseline_potential="zbl")
    dyn_cfg = DynamicsConfig(trusted_directories=[], project_root=str(Path.cwd()))
    md_engine = MDInterface(dyn_cfg, sys_cfg)
    eon_engine = EONWrapper(dyn_cfg, sys_cfg)

    ora_cfg = OracleConfig()
    oracle = DFTManager(ora_cfg)

    trn_cfg = TrainerConfig(trusted_directories=[])
    trainer = PacemakerWrapper(trn_cfg)

    gen_cfg = StructureGeneratorConfig()
    generator = StructureGenerator(gen_cfg)

    assert isinstance(md_engine, AbstractDynamics)
    assert isinstance(eon_engine, AbstractDynamics)
    assert isinstance(oracle, AbstractOracle)
    assert isinstance(trainer, AbstractTrainer)
    assert isinstance(generator, AbstractGenerator)


def test_dynamics_halt_interrupt() -> None:
    from src.core.exceptions import DynamicsHaltInterrupt

    msg = "Simulation halted due to high uncertainty"
    exc = DynamicsHaltInterrupt(msg)

    assert isinstance(exc, Exception)
    assert str(exc) == msg

def test_interface_target() -> None:
    from src.domain_models.config import InterfaceTarget
    target = InterfaceTarget(element1="FePt", element2="MgO", face1="Fe", face2="Mg")
    assert target.element1 == "FePt"
    assert target.element2 == "MgO"
    assert target.face1 == "Fe"
    assert target.face2 == "Mg"

    with pytest.raises(ValidationError):
        InterfaceTarget(element1="Fe", element2="MgO", unexpected="field") # type: ignore[call-arg]
