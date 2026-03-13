from pathlib import Path

import pytest
from src.core.orchestrator import ActiveLearningOrchestrator
from src.generators.structure_generator import StructureGenerator
from src.oracles.dft_oracle import DFTOracle

from src.domain_models.config import (
    DynamicsConfig,
    OracleConfig,
    PipelineConfig,
    SystemConfig,
    TrainerConfig,
)
from src.dynamics.dynamics_engine import DynamicsEngine
from src.trainers.ace_trainer import ACETrainer
from src.validators.validator import Validator


@pytest.fixture
def mock_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        system=SystemConfig(elements=["Ti", "O"]),
        oracle=OracleConfig(k_spacing=0.05, pseudo_paths={}),
        trainer=TrainerConfig(ace_max_degree=2, lj_baseline_params={}),
        dynamics=DynamicsConfig(),
    )


def test_pipeline_skeleton(tmp_path: Path, mock_pipeline_config: PipelineConfig) -> None:
    # Setup test workspace
    import os

    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    # Initialize components
    generator = StructureGenerator(mock_pipeline_config.system)
    oracle = DFTOracle(mock_pipeline_config.oracle)
    trainer = ACETrainer(mock_pipeline_config.trainer)
    dynamics = DynamicsEngine(mock_pipeline_config.dynamics)
    validator = Validator()

    orchestrator = ActiveLearningOrchestrator(
        config=mock_pipeline_config,
        generator=generator,
        oracle=oracle,
        trainer=trainer,
        dynamics=dynamics,
        validator=validator,
    )

    # Create the potentials directory locally in tmp_path
    Path("potentials").mkdir(parents=True, exist_ok=True)

    status = orchestrator.run_cycle()

    # Based on the dummy strategy (max temp 600) and threshold > 500 in dynamics engine, it halts and loops.
    assert status == "CYCLE_COMPLETE", f"Expected cycle to complete, got {status}"

    # Ensure potential was generated
    assert Path("potential.yace").exists()

    # Teardown
    os.chdir(original_cwd)
