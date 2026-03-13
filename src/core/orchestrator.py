import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from ase import Atoms

from src.domain_models.config import ProjectConfig
from src.dynamics.dynamics_engine import MDInterface
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine
from src.generators.structure_generator import StructureGenerator
from src.oracles.dft_oracle import DFTManager
from src.trainers.ace_trainer import PacemakerWrapper
from src.validators.validator import Validator


class Orchestrator:
    """Core logic to manage state transitions in the active learning loop."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.md_engine = MDInterface(config.dynamics, config.system)
        self.oracle = DFTManager(config.oracle)
        self.trainer = PacemakerWrapper(config.trainer)
        self.validator = Validator(config.validator)
        self.policy_engine = AdaptiveExplorationPolicyEngine(config.policy)
        self.structure_generator = StructureGenerator(config.structure_generator)
        self.iteration = 0

    def get_latest_potential(self) -> Path | None:
        """Finds the most recent valid generation potential."""
        pot_dir = self.config.project_root / "potentials"
        if not pot_dir.exists():
            return None

        # Glob generation_XXX.yace
        files = list(pot_dir.glob("generation_*.yace"))
        if not files:
            return None

        try:
            latest_pot = max(files).resolve()

            # Directory traversal vulnerability fix
            if not latest_pot.is_relative_to(pot_dir.resolve()):
                logging.error("Path traversal attempt detected resolving potential files.")
                return None

            # Validate format strictly to ensure file integrity
            with Path.open(latest_pot) as f:
                content = f.read(100)
        except (ValueError, OSError):
            logging.exception("Failed to load or validate latest potential")
            return None
        else:
            if "elements" not in content and "version" not in content:
                logging.warning(f"Potential {latest_pot} appears corrupted or invalid.")
                return None
            return latest_pot

    def run_cycle(self) -> str | None:
        """Runs one full loop: Exploration -> Selection -> DFT -> Update -> Resume."""
        logging.info(f"Starting iteration {self.iteration}")

        current_pot = self.get_latest_potential()
        if current_pot is None:
            logging.info(
                "No initial potential found. Starting cold-start exploration (Baseline only)."
            )

        # Build directory mapping
        base_dir = self.config.project_root / "active_learning"
        base_dir.mkdir(parents=True, exist_ok=True)

        work_dir = base_dir / f"iter_{self.iteration:03d}"
        # For atomic transactions, work in a fully isolated temporary space
        # and only map to the final `work_dir` if everything passes smoothly
        import tempfile

        cycle_successful = False
        tmp_work_dir = Path(tempfile.mkdtemp(dir=str(base_dir)))

        try:
            # 1. EXPLORATION
            halt_info = self.md_engine.run_exploration(
                potential=current_pot,
                work_dir=tmp_work_dir / "md_run",
            )

            if not halt_info.get("halted", False):
                logging.info("MD completed without high uncertainty. Converged.")
                cycle_successful = True
                return "CONVERGED"

            logging.warning("Halt triggered by uncertainty watchdog!")

            # 2. DETECTION & SELECTION
            high_gamma_atoms = self.md_engine.extract_high_gamma_structures(
                dump_file=halt_info["dump_file"],
                threshold=self.config.dynamics.uncertainty_threshold,
            )

            def candidate_generator() -> Iterator[list[Atoms]]:
                for s0 in high_gamma_atoms:
                    candidates = self.structure_generator.generate_local_candidates(s0, n=20)
                    yield self.trainer.select_local_active_set(candidates, anchor=s0, n=5)

            # 3. LABELING (DFT Oracle) & 4. TRAINING
            has_new_data = False
            data_dir = self.config.project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = data_dir / "accumulated.extxyz"

            for i, batch in enumerate(candidate_generator()):
                batch_calc_dir = tmp_work_dir / f"dft_calc_batch_{i}"
                new_data = self.oracle.compute_batch(batch, batch_calc_dir)
                if new_data:
                    self.trainer.update_dataset(new_data, dataset_path=dataset_path)
                    has_new_data = True

            if not has_new_data:
                logging.error("No valid data obtained from DFT.")
                return "ERROR"

            new_pot_path = self.trainer.train(
                dataset=dataset_path,
                initial_potential=current_pot,
                output_dir=tmp_work_dir / "training",
            )

            # 5. VALIDATION
            validation_result = self.validator.validate(new_pot_path)
            if not validation_result.passed:
                logging.error(f"Validation failed: {validation_result.reason}")
                return "VALIDATION_FAILED"

            # 6. DEPLOYMENT (Atomic transaction)
            self.iteration += 1
            pot_dir = self.config.project_root / "potentials"
            pot_dir.mkdir(parents=True, exist_ok=True)
            final_dest = pot_dir / f"generation_{self.iteration:03d}.yace"

            # Atomically move temporary workspace to final iteration map
            # First move the potential out
            shutil.copy(tmp_work_dir / "training" / "output_potential.yace", final_dest)

            # Then map the transaction completely
            # Use shutil.move for reliable cross-fs/dir merge semantics
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            shutil.move(str(tmp_work_dir), str(work_dir))

            cycle_successful = True
            return str(final_dest)
        finally:
            if not cycle_successful and tmp_work_dir.exists():
                logging.warning(f"Cleaning up partial state due to failure: {tmp_work_dir}")
                shutil.rmtree(tmp_work_dir, ignore_errors=True)
