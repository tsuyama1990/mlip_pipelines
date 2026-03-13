import pytest
from pydantic import ValidationError

from src.domain_models.config import (
    DynamicsConfig,
    OracleConfig,
    PipelineConfig,
    SystemConfig,
    TrainerConfig,
)
from src.domain_models.dtos import (
    ExplorationStrategy,
    HaltEvent,
    ValidationScore,
)


def test_system_config_valid() -> None:
    config = SystemConfig(elements=["Fe", "Pt"], mass={"Fe": 55.845, "Pt": 195.084})
    assert config.elements == ["Fe", "Pt"]
    assert config.mass == {"Fe": 55.845, "Pt": 195.084}

def test_system_config_invalid_extra() -> None:
    with pytest.raises(ValidationError):
        SystemConfig(elements=["Fe"], extra_param="invalid")  # type: ignore[call-arg]

def test_oracle_config_valid() -> None:
    config = OracleConfig(
        k_spacing=0.03,
        pseudo_paths={"Fe": "fe.upf"},
    )
    assert config.mixing_beta == 0.7
    assert config.smearing == 0.01

def test_trainer_config_valid() -> None:
    config = TrainerConfig(
        ace_max_degree=3,
        lj_baseline_params={"Fe-Pt": 1.0}
    )
    assert config.ace_max_degree == 3

def test_dynamics_config_valid() -> None:
    config = DynamicsConfig()
    assert config.gamma_threshold == 5.0

def test_pipeline_config_valid() -> None:
    config = PipelineConfig(
        system=SystemConfig(elements=["Fe", "Pt"]),
        oracle=OracleConfig(k_spacing=0.05, pseudo_paths={}),
        trainer=TrainerConfig(ace_max_degree=2, lj_baseline_params={}),
        dynamics=DynamicsConfig(),
    )
    assert config.system.elements == ["Fe", "Pt"]

def test_pipeline_config_invalid() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(
            system=SystemConfig(elements=["Fe", "Pt"]),
            oracle=OracleConfig(k_spacing=0.05, pseudo_paths={}),
            trainer=TrainerConfig(ace_max_degree=2, lj_baseline_params={}),
            # Missing dynamics
        ) # type: ignore[call-arg]

def test_exploration_strategy_valid() -> None:
    strategy = ExplorationStrategy(
        r_md_mc=10.0,
        t_schedule=[300.0, 600.0],
        n_defects=5,
        strain_range=0.1,
    )
    assert strategy.n_defects == 5

def test_halt_event_valid() -> None:
    event = HaltEvent(
        timestep=1000,
        max_gamma=5.5,
        triggering_atoms_indices=[0, 5],
        halt_structure="mock_atoms",
    )
    assert event.timestep == 1000

def test_validation_score_valid() -> None:
    score = ValidationScore(
        rmse_energy=0.5,
        rmse_force=0.03,
        phonon_stable=True,
        born_criteria_passed=True,
    )
    assert score.phonon_stable is True
