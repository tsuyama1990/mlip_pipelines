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

from src.core import AbstractDynamics, AbstractGenerator, AbstractTrainer, BaseOracle
from src.core.exceptions import DynamicsHaltInterrupt, OracleConvergenceError
from src.domain_models.config import ProjectConfig
from src.domain_models.dtos import ExplorationStrategy, MaterialFeatures
from src.dynamics.dynamics_engine import MDInterface
from src.dynamics.eon_wrapper import EONWrapper
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine
from src.generators.structure_generator import StructureGenerator
from src.oracles.dft_oracle import DFTManager
from src.oracles.mace_manager import MACEManager
from src.oracles.tiered_oracle import TieredOracle
from src.trainers.ace_trainer import PacemakerWrapper
from src.validators.reporter import Reporter
from src.validators.validator import Validator


class Orchestrator:
    """Core logic to manage state transitions in the active learning loop."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.md_engine: AbstractDynamics = MDInterface(config.dynamics, config.system)
        self.eon_engine: AbstractDynamics = EONWrapper(config.dynamics, config.system)

        fallback_oracle = DFTManager(config.oracle)
        if config.loop_strategy.use_tiered_oracle:
            primary_oracle = MACEManager(config)
            self.oracle: BaseOracle = TieredOracle(
                primary_oracle=primary_oracle,
                fallback_oracle=fallback_oracle,
                threshold=config.loop_strategy.thresholds.threshold_call_dft,
            )
        else:
            self.oracle = fallback_oracle

        self.trainer: AbstractTrainer = PacemakerWrapper(config.trainer)
        self.validator = Validator(config.validator)
        self.reporter = Reporter()
        self.policy_engine = AdaptiveExplorationPolicyEngine(config.policy)
        self.structure_generator: AbstractGenerator = StructureGenerator(config.structure_generator)

        from src.core.checkpoint import CheckpointManager
        from src.trainers.finetune_manager import FinetuneManager

        # Ensure we have a database path
        db_path = config.project_root / ".ac_cdd" / "checkpoint.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint = CheckpointManager(db_path)

        self.finetune_manager = FinetuneManager(config.trainer)

        # In a real active learning loop, iteration is managed by state.
        state_iter = self.checkpoint.get_state("CURRENT_ITERATION")
        if state_iter is not None:
            self.iteration = int(state_iter)
        else:
            self.resume_state()

    def resume_state(self) -> None:
        """Scans the potentials directory to find the highest completed iteration."""
        import re
        import shutil

        pot_dir = self.config.project_root / "potentials"
        if not pot_dir.exists():
            self.iteration = 0
            return

        files = list(pot_dir.glob("generation_*.yace"))
        if not files:
            self.iteration = 0
            return

        max_iter = 0
        for f in files:
            if not f.name.startswith("generation_") or not f.name.endswith(".yace"):
                continue
            match = re.search(r"generation_(\d+)\.yace", f.name)
            if match:
                max_iter = max(max_iter, int(match.group(1)))

        self.iteration = max_iter
        logging.info(f"Resumed state from generation_{self.iteration:03d}.yace")

        # Clean up orphaned tmp directories
        al_dir = self.config.project_root / "active_learning"
        if al_dir.exists():
            for tmp_dir in al_dir.glob("tmp*"):
                if tmp_dir.is_dir():
                    logging.info(f"Cleaning up orphaned temporary directory: {tmp_dir}")
                    shutil.rmtree(tmp_dir, ignore_errors=True)

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
        try:
            strategy = self._decide_exploration_strategy()
            halt_info = self._execute_exploration(strategy, current_pot, tmp_work_dir)
            return self._detect_halt(halt_info)
        except DynamicsHaltInterrupt:
            logging.exception("Exploration halted due to configured uncertainty thresholds.")
            raise
        except OracleConvergenceError:
            logging.exception("Oracle convergence failed during initial setup/exploration.")
            raise
        except Exception as e:
            logging.exception("An unexpected critical failure occurred during exploration.")
            # Ensure proper resource cleanup occurs even on unexpected generic exceptions
            import shutil

            if tmp_work_dir.exists():
                shutil.rmtree(str(tmp_work_dir), ignore_errors=True)

            msg = "Critical infrastructure failure during exploration."
            raise RuntimeError(msg) from e

    def _select_candidates(self, halt_info: dict[str, Any]) -> Iterator[list[Atoms]]:
        dump_file = halt_info.get("dump_file")
        if not dump_file:
            logging.error("No dump file provided in halt_info.")
            yield from []
            return

        dump_path = Path(dump_file)
        if not dump_path.exists() or not dump_path.is_file():
            logging.error(f"Dump file missing or invalid: {dump_path}")
            yield from []
            return

        if halt_info.get("is_kmc"):
            import ase.io

            high_gamma_atoms = [ase.io.read(dump_path)]
            if not isinstance(high_gamma_atoms[0], Atoms):
                high_gamma_atoms = [high_gamma_atoms[0][0]]
        else:
            # Cast down to MDInterface because extract_high_gamma_structures is MD-specific
            from src.dynamics.dynamics_engine import MDInterface

            md_engine = self.md_engine
            if isinstance(md_engine, MDInterface):
                high_gamma_atoms = md_engine.extract_high_gamma_structures(
                    dump_file=dump_path,
                    threshold=self.config.dynamics.uncertainty_threshold,
                )
            else:
                high_gamma_atoms = []

        for s0 in high_gamma_atoms:
            candidates = self.structure_generator.generate_local_candidates(s0, n=20)
            yield self.trainer.select_local_active_set(list(candidates), anchor=s0, n=5)

    def _compute_dft_and_update_dataset(
        self,
        candidate_generator: Iterator[list[Atoms]],
        tmp_work_dir: Path,
        dataset_path: Path,
    ) -> bool:
        has_new_data = False
        for i, batch in enumerate(candidate_generator):
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

    def _secure_copy_potential(
        self, src_pot: Path, pot_dir: Path, iteration: int, tmp_work_dir: Path
    ) -> Path:
        import hashlib
        import os
        import re
        import tempfile
        import fcntl

        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(src_pot.parent), check_exists=False)
        _secure_resolve_and_validate_dir(str(pot_dir), check_exists=False)
        _secure_resolve_and_validate_dir(str(tmp_work_dir), check_exists=False)

        if not re.match(r"^[a-zA-Z0-9_]+\.yace$", src_pot.name) or src_pot.name.startswith("-"):
            msg = "Source potential file must have a valid .yace filename format"
            raise ValueError(msg)

        # Single atomic resolution of everything to prevent TOCTOU
        try:
            resolved_src_str = os.path.realpath(str(src_pot.resolve(strict=True)))
            resolved_src = Path(resolved_src_str)

            allowed_src_dir_str = os.path.realpath(str(tmp_work_dir.resolve(strict=True)))
            allowed_src_dir = Path(allowed_src_dir_str)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                msg = "Source potential file missing or invalid"
                raise FileNotFoundError(msg) from e
            msg = f"Failed to resolve path: {e}"
            raise ValueError(msg) from e

        if not resolved_src.is_relative_to(allowed_src_dir):
            msg = f"Security Violation: Resolved source potential {resolved_src} lies outside the trusted directory {allowed_src_dir}"
            raise ValueError(msg)

        if len(resolved_src.parts) > 50:
            msg = "Security Violation: Path depth exceeds maximum allowed limit of 50."
            raise ValueError(msg)

        pot_dir.mkdir(parents=True, exist_ok=True)
        resolved_pot_dir_str = os.path.realpath(str(pot_dir.resolve()))
        resolved_pot_dir = Path(resolved_pot_dir_str)
        final_dest = resolved_pot_dir / f"generation_{iteration:03d}.yace"

        # Secure cross-filesystem atomic copy with streaming hash avoiding TOCTOU attacks
        try:
            fd, tmp_path_str = tempfile.mkstemp(dir=str(resolved_pot_dir), prefix=".tmp_pot_")
            tmp_path = Path(tmp_path_str)
            try:
                sha256_src = hashlib.sha256()
                sha256_dest = hashlib.sha256()

                # Open source file with O_RDONLY and O_NOFOLLOW to ensure we're reading exactly the validated file
                src_fd = os.open(str(resolved_src), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))

                # Apply advisory lock on the source to prevent modification while reading
                fcntl.flock(src_fd, fcntl.LOCK_SH)

                try:
                    # Validate size under lock
                    st = os.fstat(src_fd)
                    max_size = getattr(self.config.trainer, "max_potential_size", 104857600)
                    if st.st_size > max_size:
                        msg = f"Source potential file exceeds maximum allowed size ({max_size} bytes)"
                        raise ValueError(msg)

                    with os.fdopen(fd, "wb") as f_out, os.fdopen(src_fd, "rb") as f_in:
                        for chunk in iter(lambda: f_in.read(8192), b""):
                            sha256_src.update(chunk)
                            f_out.write(chunk)
                finally:
                    # Unlock happens automatically when fd is closed or process exits, but it's good practice
                    with __import__("contextlib").suppress(Exception):
                        fcntl.flock(src_fd, fcntl.LOCK_UN)

                expected_hash = sha256_src.hexdigest()

                # Hash destination to verify integrity
                with Path.open(tmp_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_dest.update(chunk)

                if sha256_dest.hexdigest() != expected_hash:
                    msg = "File integrity check failed after copying."
                    raise RuntimeError(msg)

                # Atomic replace on the same filesystem
                tmp_path.replace(final_dest)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
        except Exception as e:
            msg = f"Failed to securely copy potential: {e}"
            raise RuntimeError(msg) from e

        return final_dest
    def _copy_potential(self, tmp_work_dir: Path, pot_dir: Path, iteration: int) -> Path:
        src_pot = tmp_work_dir / "training" / "output_potential.yace"
        return self._secure_copy_potential(src_pot, pot_dir, iteration, tmp_work_dir)

    def _validate_tmp_directories(self, tmp_work_dir: Path) -> None:
        from src.domain_models.config import _secure_resolve_and_validate_dir
        _secure_resolve_and_validate_dir(str(tmp_work_dir), check_exists=False)

        expected_dirs = ["training"]
        if not (tmp_work_dir / "md_run").exists() and not (tmp_work_dir / "kmc_run").exists():
            (tmp_work_dir / "md_run").mkdir(parents=True, exist_ok=True)
        for expected in expected_dirs:
            (tmp_work_dir / expected).mkdir(parents=True, exist_ok=True)

    def _swap_directories(self, tmp_work_dir: Path, work_dir: Path) -> None:
        from src.domain_models.config import _secure_resolve_and_validate_dir
        import os
        import time
        import fcntl
        import shutil
        import tempfile

        _secure_resolve_and_validate_dir(str(tmp_work_dir), check_exists=False)
        _secure_resolve_and_validate_dir(str(work_dir), check_exists=False)

        # Single atomic resolution of parent to ensure the target is secure and exists
        try:
            work_parent_str = os.path.realpath(str(work_dir.parent.resolve(strict=True)))
            work_parent = Path(work_parent_str)
            final_dest = work_parent / work_dir.name
        except Exception as e:
            msg = f"Failed to securely resolve working directory parent: {e}"
            raise RuntimeError(msg) from e

        lock_name = getattr(self.config.loop_strategy, "swap_lock_file", ".swap.lock")
        lock_file = work_parent / lock_name

        lock_acquired = False
        timeout = 60.0
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                f_lock = Path.open(lock_file, "w")
                fcntl.flock(f_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
                break
            except (BlockingIOError, OSError):
                f_lock.close()
                time.sleep(1.0)

        if not lock_acquired:
            msg = f"Directory swap failed to acquire lock after {timeout} seconds"
            raise RuntimeError(msg)

        try:
            # Atomic rename on the same filesystem is safest
            # Create a true temp directory in the same parent folder
            temp_swap_str = tempfile.mkdtemp(dir=str(work_parent))
            temp_swap = Path(temp_swap_str)
            new_work = temp_swap / "new_work"

            try:
                # Secure copy into the isolated temp dir
                shutil.copytree(str(tmp_work_dir), str(new_work))

                # Simple rename
                if final_dest.exists():
                    backup_dest = temp_swap / "backup"
                    # Rename is atomic
                    os.rename(str(final_dest), str(backup_dest))
                    try:
                        os.rename(str(new_work), str(final_dest))
                    except Exception:
                        # Rollback atomically
                        if final_dest.exists():
                            # If new_work got there, move it away first
                            os.rename(str(final_dest), str(temp_swap / "failed_new_work"))
                        os.rename(str(backup_dest), str(final_dest))
                        raise
                else:
                    os.rename(str(new_work), str(final_dest))
            finally:
                shutil.rmtree(str(temp_swap), ignore_errors=True)
                # Clean up the original tmp_work_dir
                shutil.rmtree(str(tmp_work_dir), ignore_errors=True)
        finally:
            fcntl.flock(f_lock, fcntl.LOCK_UN)
            f_lock.close()
            import contextlib
            with contextlib.suppress(OSError):
                lock_file.unlink()
    def _manage_directories(self, tmp_work_dir: Path, work_dir: Path) -> None:
        self._validate_tmp_directories(tmp_work_dir)
        self._swap_directories(tmp_work_dir, work_dir)

    def _resume_md_engine(self, final_dest: Path, work_dir: Path) -> None:
        from src.dynamics.dynamics_engine import MDInterface

        md_engine = self.md_engine
        if isinstance(md_engine, MDInterface):
            md_engine.resume(
                potential=final_dest,
                restart_dir=work_dir / "md_run",
                work_dir=work_dir / "resume_run",
            )

    def _finalize_validation(self, new_pot_path: Path) -> bool:
        if not new_pot_path.exists() or not new_pot_path.is_file():
            msg = f"New potential path is invalid or missing: {new_pot_path}"
            raise FileNotFoundError(msg)
        return self._validate_potential(new_pot_path)

    def _deploy_potential(self, tmp_work_dir: Path) -> Path:
        self.iteration += 1
        pot_dir = self.config.project_root / "potentials"
        pot_dir.mkdir(parents=True, exist_ok=True)
        return self._copy_potential(tmp_work_dir, pot_dir, self.iteration)

    def _finalize_directories(self, tmp_work_dir: Path, work_dir: Path) -> None:
        try:
            self._manage_directories(tmp_work_dir, work_dir)
        except Exception:
            logging.exception("Failed to manage AL directories atomically")
            raise

    def _pre_generate_interface_target(self) -> str | None:
        if (
            not self.config.system.interface_target
            or self.iteration != self.config.system.interface_generation_iteration
        ):
            return None

        logging.info("Interface configuration detected. Pre-generating interface target.")
        try:
            from src.generators.structure_generator import StructureGenerator

            if isinstance(self.structure_generator, StructureGenerator):
                initial_struct = self.structure_generator.generate_interface(
                    self.config.system.interface_target
                )
                from ase.io import write

                work_dir_setup = (
                    self.config.project_root
                    / "active_learning"
                    / f"iter_{self.iteration:03d}"
                    / "md_run"
                )
                work_dir_setup.mkdir(parents=True, exist_ok=True)
                target_file = work_dir_setup / "initial_structure.extxyz"
                write(str(target_file), initial_struct, format="extxyz")
                logging.info(
                    f"Generated interface structure with {len(initial_struct)} atoms and saved to {target_file}"
                )
            else:
                logging.warning("Structure generator does not support generate_interface")
        except Exception:
            logging.exception("Failed to generate interface structure")
            return "ERROR"
        else:
            return None



    def _cleanup_artifacts(self, paths: list[Path]) -> None:
        """Daemon to aggressively clean up huge files to prevent HPC quota breaches."""
        import os
        import logging

        # Authorized base path for cleanups using single strict resolution
        try:
            active_learning_dir_str = os.path.realpath(str(
                (self.config.project_root / "active_learning").resolve(strict=True)
            ))
            active_learning_dir = Path(active_learning_dir_str)
        except Exception as e:
            logging.error(f"Failed to resolve active_learning directory for cleanup: {e}")
            return

        allowed_extensions = getattr(
            self.config.loop_strategy,
            "allowed_cleanup_extensions",
            {".dat", ".wfc", ".lammps", ".yace"},
        )
        size_threshold = getattr(self.config.loop_strategy, "cleanup_size_threshold", 10240)

        failure_count = 0

        for path in paths:
            try:
                if not path.exists():
                    continue

                # Resolve ALL symlinks before checking anything
                canonical_str = os.path.realpath(str(path.resolve(strict=True)))
                canonical_path = Path(canonical_str)

                if not canonical_path.is_file():
                    continue

                # Extension whitelist validation (checking actual file type vs suffix spoofing)
                if (
                    canonical_path.suffix not in allowed_extensions
                    and canonical_path.name != "dump.lammps"
                ):
                    logging.warning(
                        f"Validation Error: Refusing to delete file with unauthorized extension: {canonical_str}"
                    )
                    continue

                # Ensure path is strictly within the allowed directory
                if not canonical_path.is_relative_to(active_learning_dir):
                    logging.warning(
                        f"Security Violation: Refusing to delete artifact outside active_learning directory: {canonical_str}"
                    )
                    continue

                # Ensure ownership matches the current process
                st = canonical_path.stat()
                if st.st_uid != os.getuid():
                    logging.warning(
                        f"Security Violation: Refusing to delete file not owned by current user: {canonical_str}"
                    )
                    continue

                # Protect against accidental deletion of small configuration/metadata files
                if st.st_size < size_threshold:
                    logging.warning(
                        f"Validation Error: File too small for aggressive cleanup: {canonical_str}"
                    )
                    continue

                # Proceed with deletion securely
                try:
                    canonical_path.unlink()
                    logging.info(f"Cleaned up large artifact securely: {canonical_str}")
                except OSError as e:
                    logging.warning(f"OS error during file deletion for {canonical_str}: {e}")
                    failure_count += 1
            except Exception as e:
                logging.warning(f"Failed to securely cleanup artifact {path}: {e}")
                failure_count += 1

            if failure_count > 3:
                msg = "Exceeded maximum allowed cleanup failures, potentially indicating a malicious file locking DOS or persistent filesystem error."
                raise RuntimeError(msg)
    def run_cycle(self) -> str | None:
        """Runs the 4-phase Hierarchical Distillation Workflow infinitely or bounded."""
        import tempfile


        max_iters = getattr(self.config.loop_strategy, "max_iterations", 9999999)
        if max_iters is None:
            max_iters = 9999999  # effectively infinite

        while self.iteration < max_iters:
            logging.info(f"Starting iteration {self.iteration}")
            self.checkpoint.set_state("CURRENT_ITERATION", self.iteration)

            current_state = self.checkpoint.get_state("CURRENT_PHASE")
            if current_state is None:
                current_state = "PHASE1_DISTILLATION"
                self.checkpoint.set_state("CURRENT_PHASE", current_state)

            current_pot = self.get_latest_potential()
            base_dir: Path = self.config.project_root / "active_learning"
            base_dir.mkdir(parents=True, exist_ok=True)
            work_dir: Path = base_dir / f"iter_{self.iteration:03d}"

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

            try:
                if current_state in ("PHASE1_DISTILLATION", "PHASE2_VALIDATION"):
                    if current_state == "PHASE1_DISTILLATION":
                        logging.info("Executing Phase 1: Zero-Shot Distillation")
                        # Phase 1 is functionally achieved implicitly via TieredOracle routing
                        # and baseline potentials logic during exploration if no dataset exists.
                        # Thus we move directly to validation of the existing baseline.
                        self.checkpoint.set_state("CURRENT_PHASE", "PHASE2_VALIDATION")
                        current_state = "PHASE2_VALIDATION"

                    if current_state == "PHASE2_VALIDATION":
                        logging.info("Executing Phase 2: Validation")
                        if current_pot is not None and not self._validate_potential(current_pot):
                            return "VALIDATION_FAILED"
                        self.checkpoint.set_state("CURRENT_PHASE", "PHASE3_MD_EXPLORATION")
                        current_state = "PHASE3_MD_EXPLORATION"

                if current_state == "PHASE3_MD_EXPLORATION":
                    logging.info("Executing Phase 3: MD Exploration")
                    err = self._pre_generate_interface_target()
                    if err == "ERROR":
                        self.checkpoint.set_state("CURRENT_PHASE", "FAILED_FATAL")
                        return "ERROR"

                    with isolated_work_dir(base_dir) as tmp_work_dir:
                        try:
                            # MD Exploration is wrapped in while True in the spec, but we are inside a while True loop
                            halt_info = self._run_exploration(current_pot, tmp_work_dir)
                            if isinstance(halt_info, str):
                                return halt_info

                            # Standard completion (no halt)
                            self.iteration += 1
                            self.checkpoint.set_state("CURRENT_PHASE", "PHASE1_DISTILLATION")
                            continue

                        except DynamicsHaltInterrupt as halt_exc:
                            logging.info(f"DynamicsHaltInterrupt Caught: {halt_exc}")
                            # Phase 3 (Extraction) & Phase 4 (Finetune)
                            # 1. Parse Halt Data
                            halt_dict = {
                                "halt_type": "uncertainty",
                                "reason": str(halt_exc),
                                "dump_file": str(tmp_work_dir / "md_run" / "dump.lammps")
                            }

                            # 2. Extract Intelligent Cluster & DFT
                            candidate_generator = self._select_candidates(halt_dict)
                            new_pot_path = self._run_dft_and_train(candidate_generator, tmp_work_dir, current_pot)

                            if isinstance(new_pot_path, str):
                                return new_pot_path

                            # 3. Finetune explicitly triggering hierarchical trainers
                            if not self._validate_potential(new_pot_path):
                                return "VALIDATION_FAILED"

                            final_dest_str = str(self._deploy_potential(tmp_work_dir))
                            self._finalize_directories(tmp_work_dir, work_dir)

                            # 4. Explicit Update state to MD_RESUME
                            self.checkpoint.set_state("CURRENT_PHASE", "PHASE3_MD_RESUME")
                            current_state = "PHASE3_MD_RESUME"

                            # Cleanup Artifacts
                            heavy_files = [tmp_work_dir / f for f in ["wfc.dat", "dump.lammps", "wavefunctions.wfc"]]
                            self._cleanup_artifacts(heavy_files)

                if current_state == "PHASE3_MD_RESUME":
                    logging.info("Resuming MD Exploration")
                    # Smoothly resume the simulation
                    pot_to_resume = Path(final_dest_str) if 'final_dest_str' in locals() else self.get_latest_potential()
                    if pot_to_resume:
                        self._resume_md_engine(pot_to_resume, work_dir)

                    self.iteration += 1
                    self.checkpoint.set_state("CURRENT_PHASE", "PHASE1_DISTILLATION")
                    continue

            except Exception:
                logging.exception("Fatal exception during Active Learning Loop")
                self.checkpoint.set_state("CURRENT_PHASE", "FAILED_FATAL")
                raise

        return None
