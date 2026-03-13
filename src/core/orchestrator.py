import logging
import shutil
from pathlib import Path
from typing import Any

from src.domain_models.config import PipelineConfig
from src.dynamics.dynamics_engine import DynamicsEngine
from src.generators.adaptive_policy import AdaptivePolicy
from src.oracles.dft_oracle import DFTOracle
from src.trainers.ace_trainer import ACETrainer
from src.validators.validator import Validator


class ActiveLearningOrchestrator:
    """Core logic to manage state transitions in the active learning loop."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.md_engine = DynamicsEngine(
            self.config.lammps, self.config.otf_loop, self.config.material
        )
        self.oracle = DFTOracle(self.config.dft)
        self.trainer = ACETrainer(self.config.training)
        self.validator = Validator(self.config.validation, self.config.material)

        # Basic metadata simulation for policy init
        self.material_dna: dict[str, Any] = {"elements": self.config.material.elements}
        self.predicted_properties: dict[str, Any] = {
            "band_gap": self.config.material.band_gap,
            "melting_point": self.config.material.melting_point,
            "bulk_modulus": self.config.material.bulk_modulus,
        }
        self.policy_engine = AdaptivePolicy(self.material_dna, self.predicted_properties, self.config.policy)

        self.iteration = 1

    def get_latest_potential(self) -> Path | None:
        """Finds the most recent valid generation potential."""
        pot_dir = Path("potentials")
        if not pot_dir.exists():
            return None
        gens = list(pot_dir.glob("generation_*.yace"))
        if not gens:
            return None
        return max(gens)

    def run_cycle(self) -> str:
        """Runs one full loop: Exploration -> Selection -> DFT -> Update -> Resume."""
        logging.info(f"Starting iteration {self.iteration}")
        current_pot = self.get_latest_potential()

        # Build directory mapping
        base_dir = self.config.active_learning_dir
        work_dir = base_dir / f"iter_{self.iteration:03d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        strategy = self.policy_engine.generate_strategy()

        # 1. EXPLORATION
        halt_info = self.md_engine.run_exploration(
            potential_path=current_pot if current_pot else Path("none.yace"),
            strategy=strategy,
            work_dir=work_dir / "md_run",
        )

        if not halt_info.get("halted", False):
            logging.info("MD completed without high uncertainty. Converged.")
            return "CONVERGED"

        logging.warning("Halt triggered by uncertainty watchdog!")

        # 2. DETECTION & SELECTION
        high_gamma_atoms = self.md_engine.extract_high_gamma_structures(
            dump_file=halt_info["dump_file"],
            threshold=self.config.otf_loop.uncertainty_threshold,
        )

        selected_structures = []
        for s0 in high_gamma_atoms:
            # Create candidates properly by adding random permutations (rattle) to the anchor
            candidates = []
            for _ in range(10):
                c = s0.copy()  # type: ignore[no-untyped-call]
                c.rattle(stdev=0.05, seed=None)
                candidates.append(c)

            selected = self.trainer.select_local_active_set(candidates, anchor=s0, n=5)
            selected_structures.extend(selected)

        # 3. LABELING (DFT Oracle)
        new_data = self.oracle.compute_batch(selected_structures, work_dir / "dft_calc")
        if not new_data:
            logging.error("No valid data obtained from DFT.")
            return "ERROR"

        # 4. TRAINING
        dataset_path = self.trainer.update_dataset(new_data)
        new_pot_path = self.trainer.train(
            dataset=dataset_path,
            initial_potential=current_pot,
            output_dir=work_dir / "training",
        )

        # 5. VALIDATION
        validation_result = self.validator.validate(new_pot_path)
        if not validation_result.get("passed", False):
            logging.error(f"Validation failed: {validation_result.get('reason')}")
            return "VALIDATION_FAILED"

        # 6. DEPLOYMENT (Scale-up step to save for resumption)
        pot_dir = Path("potentials")
        pot_dir.mkdir(parents=True, exist_ok=True)
        final_dest = pot_dir / f"generation_{self.iteration:03d}.yace"
        shutil.copy(new_pot_path, final_dest)

        self.iteration += 1
        return "CONTINUE"
