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
        a=2.8665,
    )
    config = PipelineConfig(material=mat_config)

    from src.dynamics.dynamics_engine import DynamicsEngine
    from src.generators.adaptive_policy import AdaptivePolicy
    from src.oracles.dft_oracle import DFTOracle
    from src.trainers.ace_trainer import ACETrainer
    from src.validators.validator import Validator

    orchestrator = ActiveLearningOrchestrator(
        config=config,
        md_engine=DynamicsEngine(config.lammps, config.otf_loop, config.material),
        oracle=DFTOracle(config.dft),
        trainer=ACETrainer(config.training),
        validator=Validator(config.validation, config.material),
        policy_engine=AdaptivePolicy(
            {"elements": config.material.elements},
            {
                "band_gap": config.material.band_gap,
                "melting_point": config.material.melting_point,
                "bulk_modulus": config.material.bulk_modulus,
            },
            config.policy,
        ),
    )
    result = orchestrator.run_cycle()
    logging.info(f"Pipeline executed with result: {result}")


if __name__ == "__main__":
    main()
