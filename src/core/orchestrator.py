import logging
from pathlib import Path

from src.core.exceptions import MLIPPipelineError
from src.core.interfaces import (
    AbstractDynamics,
    AbstractGenerator,
    AbstractOracle,
    AbstractTrainer,
    AbstractValidator,
)
from src.domain_models.config import PipelineConfig
from src.generators.adaptive_policy import AdaptivePolicy

logger = logging.getLogger(__name__)


class ActiveLearningOrchestrator:
    def __init__(
        self,
        config: PipelineConfig,
        generator: AbstractGenerator,
        oracle: AbstractOracle,
        trainer: AbstractTrainer,
        dynamics: AbstractDynamics,
        validator: AbstractValidator,
    ) -> None:
        self.config = config
        self.generator = generator
        self.oracle = oracle
        self.trainer = trainer
        self.dynamics = dynamics
        self.validator = validator
        self.iteration = 0

    def run_cycle(self) -> str:
        """Runs a single active learning cycle."""
        logger.info(f"Starting Active Learning Cycle {self.iteration}")
        try:
            strategy = AdaptivePolicy.generate_strategy(self.config.system)

            # Use current potential
            pot_path_str = self.config.potential_path_template.format(iteration=self.iteration)
            current_pot = Path(pot_path_str)
            if not current_pot.exists():
                # For iteration 0, we might need initial training data.
                logger.info(
                    "No current potential found. Proceeding to generate initial structures."
                )
                initial_structures = self.generator.generate_initial_structures(strategy)
                logger.info(f"Generated {len(initial_structures)} initial structures.")

                computed_data = self.oracle.compute(initial_structures)
                logger.info(f"Computed DFT data for {len(computed_data)} structures.")

                current_pot = self.trainer.train(computed_data)
                logger.info(f"Initial potential trained at {current_pot}")

            # 1. Exploration
            halt_info = self.dynamics.run_exploration(current_pot, strategy)

            if halt_info is None:
                logger.info("Dynamics completed without high uncertainty. Converged.")
                return "CONVERGED"

            # 2. Selection
            logger.info("Halt triggered. Generating local candidates.")
            candidates = self.generator.generate_local_candidates(halt_info, strategy)
            selected = self.trainer.filter_active_set(candidates, anchor=halt_info.halt_structure)

            # 3. Compute (Oracle)
            logger.info(f"Computing exact forces for {len(selected)} selected structures.")
            new_data = self.oracle.compute(selected)
            if not new_data:
                logger.error("No valid data obtained from DFT.")
                return "ERROR"

            # 4. Train
            logger.info("Refining potential with new data.")
            new_pot_path = self.trainer.train(new_data)

            # 5. Validate
            logger.info("Validating new potential.")
            val_score = self.validator.validate(new_pot_path)
            if not val_score.phonon_stable:
                logger.warning("Validation failed: Phonon unstable.")

            self.iteration += 1

        except MLIPPipelineError:
            logger.exception("Pipeline error occurred")
            return "ERROR"
        except Exception:
            logger.exception("Unexpected error in Orchestrator cycle.")
            return "ERROR"
        else:
            return "CYCLE_COMPLETE"
