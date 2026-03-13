import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from src.domain_models.config import ProjectConfig
from src.domain_models.dtos import MaterialFeatures
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
        self.md_engine = MDInterface(config.dynamics, config.system)
        self.eon_engine = EONWrapper(config.dynamics, config.system)
        self.oracle = DFTManager(config.oracle)
        self.trainer = PacemakerWrapper(config.trainer)
        self.validator = Validator(config.validator)
        self.reporter = Reporter()
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

    def _run_exploration(
        self, current_pot: Path | None, tmp_work_dir: Path
    ) -> dict[str, Any] | str:
        # Deduce features to get policy
        features = MaterialFeatures(elements=self.config.system.elements)
        from src.domain_models.dtos import ExplorationStrategy

        try:
            strategy = self.policy_engine.decide_policy(features)
        except (ValueError, TypeError, KeyError) as e:
            logging.warning(f"Policy engine parameter calculation failed: {e}. Falling back to default MD strategy.")
            strategy = ExplorationStrategy(
                md_mc_ratio=0.0,
                t_max=300.0,
                n_defects=0.0,
                strain_range=0.0,
                policy_name="Fallback Standard",
            )
        except RuntimeError as e:
            msg = f"Critical infrastructure failure in policy engine execution: {e}"
            logging.exception(msg)
            raise RuntimeError(msg) from e

        if strategy.md_mc_ratio > 0.0:
            logging.info(f"Running kMC (EON) exploration due to strategy {strategy.policy_name}")
            halt_info = self.eon_engine.run_kmc(
                potential=current_pot,
                work_dir=tmp_work_dir / "kmc_run",
            )
        else:
            logging.info(f"Running MD exploration due to strategy {strategy.policy_name}")
            halt_info = self.md_engine.run_exploration(
                potential=current_pot,
                work_dir=tmp_work_dir / "md_run",
            )

        if not halt_info.get("halted", False):
            logging.info("Exploration completed without high uncertainty. Converged.")
            return "CONVERGED"

        logging.warning("Halt triggered by uncertainty watchdog!")
        return halt_info

    def _select_candidates(self, halt_info: dict[str, Any]) -> Iterator[list[Atoms]]:
        if halt_info.get("is_kmc"):
            import ase.io

            high_gamma_atoms = [ase.io.read(halt_info["dump_file"])]
            if not isinstance(high_gamma_atoms[0], Atoms):
                high_gamma_atoms = [high_gamma_atoms[0][0]]  # type: ignore[index]
        else:
            high_gamma_atoms = self.md_engine.extract_high_gamma_structures(
                dump_file=halt_info["dump_file"],
                threshold=self.config.dynamics.uncertainty_threshold,
            )

        for s0 in high_gamma_atoms:
            candidates = self.structure_generator.generate_local_candidates(s0, n=20)
            yield self.trainer.select_local_active_set(candidates, anchor=s0, n=5)

    def _run_dft_and_train(
        self,
        candidate_generator: Iterator[list[Atoms]],
        tmp_work_dir: Path,
        current_pot: Path | None,
    ) -> Path | str:
        has_new_data = False
        data_dir = self.config.project_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = data_dir / "accumulated.extxyz"

        for i, batch in enumerate(candidate_generator):
            batch_calc_dir = tmp_work_dir / f"dft_calc_batch_{i}"
            new_data = self.oracle.compute_batch(batch, batch_calc_dir)
            if new_data:
                self.trainer.update_dataset(new_data, dataset_path=dataset_path)
                has_new_data = True

        if not has_new_data:
            logging.error("No valid data obtained from DFT.")
            return "ERROR"

        return self.trainer.train(
            dataset=dataset_path,
            initial_potential=current_pot,
            output_dir=tmp_work_dir / "training",
        )

    def _validate_and_deploy(self, new_pot_path: Path, tmp_work_dir: Path, work_dir: Path) -> str:  # noqa: PLR0915
        validation_result = self.validator.validate(new_pot_path)

        # decoupled reporter usage
        report_path = new_pot_path.parent / "validation_report.html"
        self.reporter.generate_html_report(validation_result, report_path)

        if not validation_result.passed:
            logging.error(f"Validation failed: {validation_result.reason}")
            return "VALIDATION_FAILED"

        self.iteration += 1
        pot_dir = self.config.project_root / "potentials"
        pot_dir.mkdir(parents=True, exist_ok=True)
        final_dest = pot_dir / f"generation_{self.iteration:03d}.yace"

        src_pot = tmp_work_dir / "training" / "output_potential.yace"
        if not src_pot.is_file() or not src_pot.resolve(strict=True).is_relative_to(
            tmp_work_dir.resolve(strict=True)
        ):
            msg = "Source potential file missing or invalid"
            raise FileNotFoundError(msg)

        if not src_pot.name.endswith(".yace"):
            msg = "Source potential file must have .yace extension"
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

        # Canonicalize the final destination path entirely to prevent traversing out of potentials
        final_dest_resolved = final_dest.resolve(strict=False)
        resolved_pot_dir = pot_dir.resolve(strict=True)
        if not final_dest_resolved.is_relative_to(resolved_pot_dir):
            msg = "final_dest is outside the expected potentials directory"
            raise ValueError(msg)

        import tempfile

        # Atomic file replacement for the trained potential rather than non-atomic copy
        # Use os.replace for guaranteed platform atomic behavior, moving via temp file if cross-device
        try:
            src_pot.resolve(strict=True).replace(final_dest_resolved)
        except OSError:
            import shutil
            shutil.move(str(src_pot.resolve(strict=True)), str(final_dest_resolved))

        # Implement directory content validation
        expected_dirs = ["training"]
        # Depending on strategy, either md_run or kmc_run should exist
        if not (tmp_work_dir / "md_run").exists() and not (tmp_work_dir / "kmc_run").exists():
            msg = "Missing required exploration directory (md_run or kmc_run)"
            raise FileNotFoundError(msg)

        for expected in expected_dirs:
            if not (tmp_work_dir / expected).exists():
                msg = f"Missing expected output directory: {expected}"
                raise FileNotFoundError(msg)

        # Use atomic rename to replace the target directory entirely, preventing race conditions

        # If work_dir exists, atomic replace is trickier cross-platform for directories,
        # but os.replace works for empty/replaced structures on POSIX
        if work_dir.exists():
            # Atomically move old work_dir out of the way before replacing
            backup_dir = Path(tempfile.mkdtemp(dir=str(work_dir.parent)))
            work_dir.replace(backup_dir)
            import shutil
            shutil.rmtree(str(backup_dir), ignore_errors=True)

        tmp_work_dir.replace(work_dir.resolve(strict=False))

        self.md_engine.resume(
            potential=final_dest,
            restart_dir=work_dir / "md_run",
            work_dir=work_dir / "resume_run",
        )

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
        import contextlib
        import os
        import tempfile
        from collections.abc import Generator

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
