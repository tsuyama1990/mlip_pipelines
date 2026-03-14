import contextlib
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from src.core import AbstractDynamics, AbstractGenerator, AbstractOracle, AbstractTrainer
from src.domain_models.config import ProjectConfig
from src.domain_models.dtos import ExplorationStrategy, MaterialFeatures
from src.dynamics.dynamics_engine import MDInterface
from src.dynamics.eon_wrapper import EONWrapper
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine
from src.generators.structure_generator import StructureGenerator
from src.oracles.dft_oracle import DFTManager
from src.trainers.ace_trainer import PacemakerWrapper
from src.validators.reporter import Reporter
from src.validators.validator import Validator


class Orchestrator:
    """Core logic to manage state transitions in the active learning loop."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.md_engine: AbstractDynamics = MDInterface(config.dynamics, config.system)
        self.eon_engine: AbstractDynamics = EONWrapper(config.dynamics, config.system)
        self.oracle: AbstractOracle = DFTManager(config.oracle)
        self.trainer: AbstractTrainer = PacemakerWrapper(config.trainer)
        self.validator = Validator(config.validator)
        self.reporter = Reporter()
        self.policy_engine = AdaptiveExplorationPolicyEngine(config.policy)
        self.structure_generator: AbstractGenerator = StructureGenerator(config.structure_generator)
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
            latest_pot = max(files).resolve(strict=True)

            # Directory traversal vulnerability fix
            if not latest_pot.is_relative_to(pot_dir.resolve()):
                msg = "Invalid potential path"
                raise ValueError(msg)

            if latest_pot.is_symlink():
                msg = "Potential path cannot be a symlink"
                raise ValueError(msg)

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

    def _decide_exploration_strategy(self) -> ExplorationStrategy:
        features = MaterialFeatures(elements=self.config.system.elements)

        try:
            return self.policy_engine.decide_policy(features)
        except (ValueError, TypeError, KeyError):
            logging.warning(
                "Policy engine parameter calculation failed. Falling back to default MD strategy."
            )
            return ExplorationStrategy(
                md_mc_ratio=0.0,
                t_max=300.0,
                n_defects=0.0,
                strain_range=0.0,
                policy_name="Fallback Standard",
            )
        except RuntimeError as e:
            msg = "Critical infrastructure failure in policy engine execution."
            logging.exception(msg)
            raise RuntimeError(msg) from e

    def _execute_exploration(
        self, strategy: ExplorationStrategy, current_pot: Path | None, tmp_work_dir: Path
    ) -> dict[str, Any]:
        if strategy.md_mc_ratio > 0.0:
            logging.info(f"Running kMC (EON) exploration due to strategy {strategy.policy_name}")
            return self.eon_engine.run_exploration(
                potential=current_pot,
                work_dir=tmp_work_dir / "kmc_run",
            )

        logging.info(f"Running MD exploration due to strategy {strategy.policy_name}")
        return self.md_engine.run_exploration(
            potential=current_pot,
            work_dir=tmp_work_dir / "md_run",
        )

    def _detect_halt(self, halt_info: dict[str, Any]) -> dict[str, Any] | str:
        if not halt_info.get("halted", False):
            logging.info("Exploration completed without high uncertainty. Converged.")
            return "CONVERGED"

        logging.warning("Halt triggered by uncertainty watchdog!")
        return halt_info

    def _run_exploration(
        self, current_pot: Path | None, tmp_work_dir: Path
    ) -> dict[str, Any] | str:
        strategy = self._decide_exploration_strategy()
        halt_info = self._execute_exploration(strategy, current_pot, tmp_work_dir)
        return self._detect_halt(halt_info)

    def _select_candidates(self, halt_info: dict[str, Any]) -> Iterator[list[Atoms]]:
        if halt_info.get("is_kmc"):
            import ase.io

            high_gamma_atoms = [ase.io.read(halt_info["dump_file"])]
            if not isinstance(high_gamma_atoms[0], Atoms):
                high_gamma_atoms = [high_gamma_atoms[0][0]]
        else:
            # Cast down to MDInterface because extract_high_gamma_structures is MD-specific
            from src.dynamics.dynamics_engine import MDInterface

            md_engine = self.md_engine
            if isinstance(md_engine, MDInterface):
                high_gamma_atoms = md_engine.extract_high_gamma_structures(
                    dump_file=halt_info["dump_file"],
                    threshold=self.config.dynamics.uncertainty_threshold,
                )
            else:
                high_gamma_atoms = []

        for s0 in high_gamma_atoms:
            candidates = self.structure_generator.generate_local_candidates(s0, n=20)
            yield self.trainer.select_local_active_set(candidates, anchor=s0, n=5)

    def _compute_dft_and_update_dataset(
        self,
        candidate_generator: Iterator[list[Atoms]],
        tmp_work_dir: Path,
        dataset_path: Path,
    ) -> bool:
        has_new_data = False
        for i, batch in enumerate(candidate_generator):
            if i >= 100:
                logging.warning("Maximum batch limit reached. Stopping generator to prevent resource exhaustion.")
                break
            batch_calc_dir = tmp_work_dir / f"dft_calc_batch_{i}"
            new_data = self.oracle.compute_batch(batch, batch_calc_dir)
            if new_data:
                self.trainer.update_dataset(new_data, dataset_path=dataset_path)
                has_new_data = True
        return has_new_data

    def _train_model(
        self, dataset_path: Path, current_pot: Path | None, tmp_work_dir: Path
    ) -> Path:
        return self.trainer.train(
            dataset=dataset_path,
            initial_potential=current_pot,
            output_dir=tmp_work_dir / "training",
        )

    def _run_dft_and_train(
        self,
        candidate_generator: Iterator[list[Atoms]],
        tmp_work_dir: Path,
        current_pot: Path | None,
    ) -> Path | str:
        data_dir = self.config.project_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = data_dir / "accumulated.extxyz"

        has_new_data = self._compute_dft_and_update_dataset(
            candidate_generator, tmp_work_dir, dataset_path
        )

        if not has_new_data:
            logging.error("No valid data obtained from DFT.")
            return "ERROR"

        return self._train_model(dataset_path, current_pot, tmp_work_dir)

    def _validate_potential(self, new_pot_path: Path) -> bool:
        validation_result = self.validator.validate(new_pot_path)
        report_path = new_pot_path.parent / "validation_report.html"
        self.reporter.generate_html_report(validation_result, report_path)
        if not validation_result.passed:
            logging.error(f"Validation failed: {validation_result.reason}")
            return False
        return True

    def _copy_potential(self, tmp_work_dir: Path, pot_dir: Path, iteration: int) -> Path:
        final_dest = pot_dir / f"generation_{iteration:03d}.yace"
        src_pot = tmp_work_dir / "training" / "output_potential.yace"

        if not src_pot.is_file():
            msg = "Source potential file missing or invalid"
            raise FileNotFoundError(msg)

        if not src_pot.resolve(strict=True).is_relative_to(tmp_work_dir.resolve(strict=True)):
            msg = "Path traversal detected"
            raise ValueError(msg)

        if not re.match(r"^[a-zA-Z0-9_-]+\.yace$", src_pot.name):
            msg = "Source potential file must have a valid .yace filename format"
            raise ValueError(msg)

        with Path.open(src_pot) as f:
            content = f.read(100)

        if "elements" not in content and "version" not in content:
            msg = f"Source potential file {src_pot} does not appear to be a valid YACE format prior to copy."
            raise ValueError(msg)

        resolved_tmp = tmp_work_dir.resolve(strict=True)
        base_al_dir = (self.config.project_root / "active_learning").resolve(strict=True)
        if not resolved_tmp.is_relative_to(base_al_dir):
            msg = "tmp_work_dir is outside the expected base directory"
            raise ValueError(msg)

        final_dest_resolved = final_dest.resolve(strict=False)
        resolved_pot_dir = pot_dir.resolve(strict=True)
        if not final_dest_resolved.is_relative_to(resolved_pot_dir):
            msg = "final_dest is outside the expected potentials directory"
            raise ValueError(msg)

        fd, tmp_dest = tempfile.mkstemp(dir=str(final_dest_resolved.parent))
        os.close(fd)

        try:
            shutil.copy2(str(src_pot.resolve(strict=True)), tmp_dest)
            Path(tmp_dest).replace(final_dest_resolved)
        except Exception:
            Path(tmp_dest).unlink(missing_ok=True)
            raise

        return final_dest

    def _manage_directories(self, tmp_work_dir: Path, work_dir: Path) -> None:
        expected_dirs = ["training"]
        if not (tmp_work_dir / "md_run").exists() and not (tmp_work_dir / "kmc_run").exists():
            msg = "Missing required exploration directory (md_run or kmc_run)"
            raise FileNotFoundError(msg)

        for expected in expected_dirs:
            if not (tmp_work_dir / expected).exists():
                msg = f"Missing expected output directory: {expected}"
                raise FileNotFoundError(msg)

        if work_dir.exists():
            backup_dir = Path(tempfile.mkdtemp(dir=str(work_dir.parent)))
            try:
                work_dir.replace(backup_dir)
                tmp_work_dir.replace(work_dir.resolve(strict=False))
            except Exception:
                if backup_dir.exists() and not work_dir.exists():
                    backup_dir.replace(work_dir)
                raise
            finally:
                shutil.rmtree(str(backup_dir), ignore_errors=True)
        else:
            tmp_work_dir.replace(work_dir.resolve(strict=False))

    def _resume_md_engine(self, final_dest: Path, work_dir: Path) -> None:
        from src.dynamics.dynamics_engine import MDInterface

        md_engine = self.md_engine
        if isinstance(md_engine, MDInterface):
            md_engine.resume(
                potential=final_dest,
                restart_dir=work_dir / "md_run",
                work_dir=work_dir / "resume_run",
            )

    def _validate_and_deploy(self, new_pot_path: Path, tmp_work_dir: Path, work_dir: Path) -> str:
        if not new_pot_path.exists() or not new_pot_path.is_file():
            msg = f"New potential path is invalid or missing: {new_pot_path}"
            raise FileNotFoundError(msg)

        if not self._validate_potential(new_pot_path):
            return "VALIDATION_FAILED"

        self.iteration += 1
        pot_dir = self.config.project_root / "potentials"
        pot_dir.mkdir(parents=True, exist_ok=True)

        final_dest = self._copy_potential(tmp_work_dir, pot_dir, self.iteration)
        self._manage_directories(tmp_work_dir, work_dir)
        self._resume_md_engine(final_dest, work_dir)

        return str(final_dest)

    def run_cycle(self) -> str | None:
        """Runs one full loop: Exploration -> Selection -> DFT -> Update -> Resume."""
        logging.info(f"Starting iteration {self.iteration}")

        current_pot = self.get_latest_potential()
        if current_pot is None:
            logging.info(
                "No initial potential found. Starting cold-start exploration (Baseline only)."
            )

        base_dir: Path = self.config.project_root / "active_learning"
        base_dir.mkdir(parents=True, exist_ok=True)

        work_dir: Path = base_dir / f"iter_{self.iteration:03d}"

        # Cleanly instantiate temp dir via robust context manager mapping explicitly for resource cleanup
        import os
        import tempfile

        @contextlib.contextmanager
        def isolated_work_dir(base: Path) -> Generator[Path, None, None]:
            tmp_path = Path(tempfile.mkdtemp(dir=str(base)))
            if tmp_path.stat().st_uid == os.getuid():
                tmp_path.chmod(0o700)
            try:
                yield tmp_path
            finally:
                if tmp_path.exists():
                    shutil.rmtree(tmp_path, ignore_errors=True)

        with isolated_work_dir(base_dir) as tmp_work_dir:
            halt_info: dict[str, Any] | str = self._run_exploration(current_pot, tmp_work_dir)
            if isinstance(halt_info, str):
                return halt_info

            candidate_generator: Iterator[list[Atoms]] = self._select_candidates(halt_info)

            new_pot_path: Path | str = self._run_dft_and_train(
                candidate_generator, tmp_work_dir, current_pot
            )
            if isinstance(new_pot_path, str):
                return new_pot_path

            final_dest: str = self._validate_and_deploy(new_pot_path, tmp_work_dir, work_dir)
            if final_dest == "VALIDATION_FAILED":
                return "VALIDATION_FAILED"

            return final_dest
