import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.domain_models.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mlip-pipelines active learning orchestrator.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
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

        # Placeholder for actual orchestration startup in future cycles
        logger.info("Initialization complete. Pipeline ready to orchestrate.")
    except yaml.YAMLError:
        logger.exception("Error parsing YAML file")
        sys.exit(1)
    except Exception:
        logger.exception("Configuration Validation Error")
        sys.exit(1)


if __name__ == "__main__":
    main()
