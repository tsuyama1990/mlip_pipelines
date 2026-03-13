import logging

from src.core.orchestrator import ActiveLearningOrchestrator
from src.domain_models.config import PipelineConfig

logging.basicConfig(level=logging.INFO)


def main() -> None:
    config = PipelineConfig()
    orchestrator = ActiveLearningOrchestrator(config)
    result = orchestrator.run_cycle()
    logging.info(f"Pipeline executed with result: {result}")


if __name__ == "__main__":
    main()
