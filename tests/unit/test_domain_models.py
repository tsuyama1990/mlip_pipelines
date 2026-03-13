import pytest
from pydantic import ValidationError
from src.domain_models.config import PipelineConfig, MDConfig, ValidationConfig
from src.domain_models.dtos import ExplorationStrategy


def test_pipeline_config_defaults() -> None:
    config = PipelineConfig()
    assert config.project_name == "mlip_project"
    assert config.lammps.temperature == 300.0


def test_pipeline_config_invalid() -> None:
    with pytest.raises(ValidationError):
        MDConfig(temperature=-10.0)

    with pytest.raises(ValidationError):
        ValidationConfig(rmse_energy_threshold=0.0)


def test_exploration_strategy_defaults() -> None:
    strategy = ExplorationStrategy()
    assert strategy.r_md_mc == 0
    assert strategy.t_schedule == (300.0, 300.0, 10000)
    assert strategy.n_defects == 0
    assert strategy.strain_range == 0.0
    assert strategy.policy_type == "Standard-MD"


def test_exploration_strategy_invalid() -> None:
    with pytest.raises(ValidationError):
        ExplorationStrategy(r_md_mc=-5)

    with pytest.raises(ValidationError):
        ExplorationStrategy(n_defects=-1)

    with pytest.raises(ValidationError):
        ExplorationStrategy(strain_range=-0.1)

    with pytest.raises(ValidationError):
        ExplorationStrategy(policy_type="Invalid-Policy")  # type: ignore[arg-type]
