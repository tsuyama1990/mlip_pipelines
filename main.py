import logging

from src.core.orchestrator import ActiveLearningOrchestrator
from src.domain_models.config import PipelineConfig

logging.basicConfig(level=logging.INFO)


def main() -> None:
    from src.domain_models.config import MaterialConfig

    mat_config = MaterialConfig(
        elements=["Fe", "Pt"],
        atomic_numbers=[26, 78],
        masses=[55.845, 195.084],
        band_gap=0.0,
        melting_point=1500.0,
        bulk_modulus=180.0,
        crystal="bcc",
        a=2.8665
    )
    config = PipelineConfig(material=mat_config)
    orchestrator = ActiveLearningOrchestrator(config)
    result = orchestrator.run_cycle()
    logging.info(f"Pipeline executed with result: {result}")


if __name__ == "__main__":
    main()
