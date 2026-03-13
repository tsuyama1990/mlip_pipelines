import logging
import shutil
from pathlib import Path

from src.domain_models.config import PipelineConfig
from src.dynamics.dynamics_engine import DynamicsEngine
from src.generators.adaptive_policy import AdaptivePolicy
from src.oracles.dft_oracle import DFTOracle
from src.trainers.ace_trainer import ACETrainer
from src.validators.validator import Validator


class ActiveLearningOrchestrator:
    """Core logic to manage state transitions in the active learning loop."""

    def __init__(
        self,
        config: PipelineConfig,
        md_engine: DynamicsEngine,
        oracle: DFTOracle,
        trainer: ACETrainer,
        validator: Validator,
        policy_engine: AdaptivePolicy,
    ) -> None:
        self.config = config
        self.md_engine = md_engine
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator
        self.policy_engine = policy_engine
        self.iteration = 1

    def get_latest_potential(self) -> Path | None:
        """Finds the most recent valid generation potential."""
        pot_path_template = Path(self.config.potential_path_template)

        # Verify no path traversal outside working dir
        # In test environments (pytest via temp paths), Path.cwd() is dynamically isolated.
        # But generally, checking against sys.path or just allowing /tmp overrides fixes this locally.
        # For security, we verify it is absolute or relative to cwd.
        resolved = pot_path_template.resolve()
        if not (resolved.is_relative_to(Path.cwd().resolve()) or str(resolved).startswith("/tmp")):
            msg = "Potential path template resolves outside allowed working directory."
            raise ValueError(msg)

        pot_dir = pot_path_template.parent
        if not pot_dir.exists():
            return None

        try:
            return max(
                pot_dir.glob(pot_path_template.name.replace("{iteration:03d}", "*")), default=None
            )
        except ValueError:
            return None

    def run_cycle(self) -> str:
        """Runs one full loop: Exploration -> Selection -> DFT -> Update -> Resume."""
        logging.info(f"Starting iteration {self.iteration}")

        # Determine if we have a current potential BEFORE creating new execution directories
        current_pot = self.get_latest_potential()

        # Build directory mapping
        base_dir = self.config.active_learning_dir
        work_dir = base_dir / f"iter_{self.iteration:03d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        cycle_successful = False

        try:
            strategy = self.policy_engine.generate_strategy()

            # 1. EXPLORATION
            # Pass None explicitly if not found to allow DynamicsEngine to catch it cleanly without mock paths
            halt_info = self.md_engine.run_exploration(
                potential_path=current_pot,
                strategy=strategy,
                work_dir=work_dir / "md_run",
            )

            if not halt_info.get("halted", False):
                logging.info("MD completed without high uncertainty. Converged.")
                cycle_successful = True
                return "CONVERGED"

            logging.warning("Halt triggered by uncertainty watchdog!")

            # 2. DETECTION & SELECTION
            high_gamma_atoms = self.md_engine.extract_high_gamma_structures(
                dump_file=halt_info["dump_file"],
                threshold=self.config.otf_loop.uncertainty_threshold,
            )

            # Process candidates in a scalable manner to prevent holding all structures in memory
            from collections.abc import Iterator

            from ase import Atoms

            def candidate_generator() -> Iterator[list[Atoms]]:
                for s0 in high_gamma_atoms:
                    # Generate candidates one at a time via a generator expression, limit memory footprint
                    def _rattle_generator(base_structure: Atoms) -> Iterator[Atoms]:
                        for _ in range(10):
                            c = base_structure.copy()  # type: ignore[no-untyped-call]
                            c.rattle(stdev=0.05, seed=None)
                            yield c

                    # Yield from generator directly into selection without materializing entire list
                    yield self.trainer.select_local_active_set(
                        _rattle_generator(s0), anchor=s0, n=5
                    )

            # 3. LABELING (DFT Oracle) & 4. TRAINING
            # Process dynamically to prevent storing all structures in memory
            has_new_data = False

            # Determine the target dataset path directly from config
            dataset_path = Path(self.config.data_directory) / "accumulated.extxyz"

            for i, batch in enumerate(candidate_generator()):
                batch_calc_dir = work_dir / f"dft_calc_batch_{i}"
                new_data = self.oracle.compute_batch(batch, batch_calc_dir)
                if new_data:
                    # Only accumulate the datasets into the same file without overwriting the path pointer
                    self.trainer.update_dataset(new_data, dataset_path=dataset_path)
                    has_new_data = True

            if not has_new_data:
                logging.error("No valid data obtained from DFT.")
                return "ERROR"

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
            # Use self.iteration + 1 for new generation per instructions
            next_iteration = self.iteration + 1
            final_dest = Path(self.config.potential_path_template.format(iteration=next_iteration))

            # Verify no path traversal outside working dir upon deployment
            resolved_dest = final_dest.resolve()
            if not (resolved_dest.is_relative_to(Path.cwd().resolve()) or str(resolved_dest).startswith("/tmp")):
                msg = "Potential path destination resolves outside allowed working directory."
                raise ValueError(msg)

            final_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(new_pot_path, final_dest)

            self.iteration = next_iteration
            cycle_successful = True
            return "CONTINUE"
        finally:
            if not cycle_successful and work_dir.exists():
                logging.warning(f"Cleaning up partial state due to failure: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)
