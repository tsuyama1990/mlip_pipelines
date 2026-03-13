import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.core.orchestrator import ActiveLearningOrchestrator
from src.domain_models.config import PipelineConfig
from src.dynamics.dynamics_engine import DynamicsEngine
from src.generators.structure_generator import StructureGenerator
from src.oracles.dft_oracle import DFTOracle
from src.trainers.ace_trainer import ACETrainer
from src.validators.validator import Validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the mlip-pipelines active learning orchestrator."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"Configuration file '{config_path}' does not exist.")
        sys.exit(1)

    try:
        with config_path.open() as f:
            config_dict = yaml.safe_load(f)

        # Validate configuration using Pydantic model
        pipeline_config = PipelineConfig(**config_dict)
        logger.info("Successfully validated configuration.")
        logger.debug(pipeline_config.model_dump_json(indent=2))

        # Initialize modules
        generator = StructureGenerator(pipeline_config.system)
        oracle = DFTOracle(pipeline_config.oracle)
        trainer = ACETrainer(pipeline_config.trainer)
        dynamics = DynamicsEngine(pipeline_config.dynamics)
        validator = Validator()

        # Run Orchestrator
        orchestrator = ActiveLearningOrchestrator(
            config=pipeline_config,
            generator=generator,
            oracle=oracle,
            trainer=trainer,
            dynamics=dynamics,
            validator=validator,
        )
        logger.info("Initialization complete. Running pipeline orchestration.")
        status = orchestrator.run_cycle()
        logger.info(f"Pipeline finished with status: {status}")

    except yaml.YAMLError:
        logger.exception("Error parsing YAML file")
        sys.exit(1)
    except Exception:
        logger.exception("Configuration Validation Error")
        sys.exit(1)


if __name__ == "__main__":
    main()
