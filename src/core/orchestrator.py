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
from src.core.exceptions import DynamicsHaltInterrupt
from src.core.retry import RetryManager
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
            # max(files) will sort them lexicographically, which is correct for generation_000.yace, etc.
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

        fallback = ExplorationStrategy(
            md_mc_ratio=0.0,
            t_max=300.0,
            n_defects=0.0,
            strain_range=0.0,
            policy_name="Fallback Standard",
        )

        # Retry logic and circuit breaking is handled by RetryManager
        retry_manager = getattr(self, "_policy_retry_manager", None)
        if not retry_manager:
            retry_manager = RetryManager()
            self._policy_retry_manager = retry_manager

        try:
            strategy = retry_manager.execute_with_retry(self.policy_engine.decide_policy, features)
        except RuntimeError:
            # Reached when transient retries are exhausted or breaker is open
            logging.exception("Policy engine failed to execute. Falling back to default.")
            return fallback
        except (ValueError, TypeError, KeyError) as e:
            # Reached when the operation yields a permanent schema/validation error internally
            logging.warning(
                f"Permanent failure in policy engine parameter calculation ({e}). Falling back to default MD strategy.",
                exc_info=True,
            )
            return fallback
        except Exception as e:
            msg = f"Critical unexpected infrastructure failure in policy engine execution: {e}"
            logging.exception(msg)
            raise RuntimeError(msg) from e
        else:
            if isinstance(strategy, ExplorationStrategy):
                return strategy
            return fallback

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
        import concurrent.futures

        strategy = self._decide_exploration_strategy()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._execute_exploration, strategy, current_pot, tmp_work_dir
                )
                timeout_seconds = getattr(self.config.loop_strategy, "timeout_seconds", 86400)
                # Exploration should not take the full global timeout, we limit it.
                exploration_timeout = timeout_seconds * 0.9
                halt_info = future.result(timeout=exploration_timeout)
        except concurrent.futures.TimeoutError as e:
            executor.shutdown(wait=False, cancel_futures=True)
            msg = "Exploration execution timed out"
            raise RuntimeError(msg) from e

        return self._detect_halt(halt_info)

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

        max_dump_size = 5 * 1024 * 1024 * 1024  # 5 GB limit to prevent OOM
        if dump_path.stat().st_size > max_dump_size:
            logging.error(f"Dump file exceeds 5GB limit: {dump_path}")
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

        mace_calc = None
        if (
            hasattr(self, "oracle")
            and isinstance(self.oracle, TieredOracle)
            and hasattr(self.oracle, "primary_oracle")
        ):
            mace_calc_provider = getattr(self.oracle.primary_oracle, "_init_calculator", None)
            if callable(mace_calc_provider):
                try:
                    mace_calc = mace_calc_provider()
                except Exception as e:
                    logging.warning(f"Could not initialize MACE calculator for pre-relaxation: {e}")

        for s0 in high_gamma_atoms:
            # Phase 3: Intelligent Cutout
            # Determine target atoms dynamically based on highest gamma values
            import numpy as np

            target_atoms = []
            if "c_pace_gamma" in s0.arrays:
                gamma = s0.arrays["c_pace_gamma"]
                threshold = self.config.loop_strategy.thresholds.threshold_add_train
                target_atoms = np.where(gamma > threshold)[0].tolist()

            if not target_atoms:
                # Fallback to atom with max gamma
                if "c_pace_gamma" in s0.arrays:
                    target_atoms = [int(np.argmax(s0.arrays["c_pace_gamma"]))]
                else:
                    target_atoms = [0]

            s_base = s0
            # Use structure generator for extraction and pre-relaxation
            if hasattr(self.structure_generator, "extract_intelligent_cluster"):
                try:
                    s_base = self.structure_generator.extract_intelligent_cluster(  # type: ignore[no-untyped-call]
                        structure=s0,
                        target_atoms=target_atoms,
                        config=self.config.cutout_config,
                        mace_calc=mace_calc,
                    )
                except Exception as e:
                    logging.warning(f"Failed to extract intelligent cluster: {e}")

            candidates = self.structure_generator.generate_local_candidates(s_base, n=20)
            yield self.trainer.select_local_active_set(list(candidates), anchor=s_base, n=5)

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

        # Isolate this specific iteration's new data into a temporary batch file
        import os
        import tempfile

        from ase.io import read, write

        fd, temp_batch_str = tempfile.mkstemp(
            dir=str(tmp_work_dir), prefix="dft_batch_", suffix=".extxyz"
        )
        os.close(fd)
        temp_batch_path = Path(temp_batch_str)

        has_new_data = self._compute_dft_and_update_dataset(
            candidate_generator, tmp_work_dir, temp_batch_path
        )

        if not has_new_data:
            logging.error("No valid data obtained from DFT.")
            if temp_batch_path.exists():
                temp_batch_path.unlink()
            return "ERROR"

        # Merge with global accumulated dataset (always keep an unpruned record for potential cold-starts)
        global_dataset_path = data_dir / "accumulated.extxyz"
        if temp_batch_path.exists():
            try:
                new_data = read(str(temp_batch_path), index=":")
                if not isinstance(new_data, list):
                    new_data = [new_data]
                self.trainer.update_dataset(new_data, global_dataset_path)
            except Exception as e:
                logging.warning(f"Failed to append to global dataset: {e}")

        train_dataset_path = temp_batch_path

        # Phase 4: Incremental Update & Replay Buffer
        if (
            getattr(self.config.loop_strategy, "incremental_update", False)
            and temp_batch_path.exists()
        ):
            try:
                new_surrogate_data = read(str(temp_batch_path), index=":")
                if not isinstance(new_surrogate_data, list):
                    new_surrogate_data = [new_surrogate_data]

                history_file_path = self.config.project_root / "data" / "training_history.extxyz"
                buffer_size = getattr(self.config.loop_strategy, "replay_buffer_size", 500)

                if hasattr(self.trainer, "manage_replay_buffer"):
                    combined_data = self.trainer.manage_replay_buffer(
                        new_surrogate_data=new_surrogate_data,
                        history_file_path=history_file_path,
                        buffer_size=buffer_size,
                    )

                    fd_t, temp_train_str = tempfile.mkstemp(
                        dir=str(tmp_work_dir), prefix="train_batch_", suffix=".extxyz"
                    )
                    os.close(fd_t)
                    temp_train_path = Path(temp_train_str)
                    write(temp_train_str, combined_data, format="extxyz")
                    train_dataset_path = temp_train_path
            except Exception as e:
                logging.warning(f"Failed to process replay buffer correctly: {e}")

        return self._train_model(train_dataset_path, current_pot, tmp_work_dir)

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
        import tempfile

        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(src_pot.parent), check_exists=False)
        _secure_resolve_and_validate_dir(str(pot_dir), check_exists=False)
        _secure_resolve_and_validate_dir(str(tmp_work_dir), check_exists=False)

        if not src_pot.exists() or not src_pot.is_file():
            msg = "Source potential file missing or invalid"
            raise FileNotFoundError(msg)

        resolved_src = src_pot.resolve(strict=True)

        if not re.match(r"^[a-zA-Z0-9_-]+\.yace$", src_pot.name):
            msg = "Source potential file must have a valid .yace filename format"
            raise ValueError(msg)

        # Cross-check that the fully resolved source file is within the explicitly allowed workspace
        # to prevent complex symlink/traversal attacks where a file named 'valid.yace'
        # points to a restricted target path outside the active learning bounds
        allowed_src_dir = os.path.realpath(str(tmp_work_dir.resolve(strict=True)))
        try:
            is_valid_path = (
                os.path.commonpath([allowed_src_dir, os.path.realpath(str(resolved_src))])
                == allowed_src_dir
            )
        except ValueError:
            is_valid_path = False

        if not is_valid_path:
            msg = f"Security Violation: Resolved source potential {resolved_src} lies outside the trusted directory {allowed_src_dir}"
            raise ValueError(msg)

        if len(resolved_src.parts) > 50:
            msg = "Security Violation: Path depth exceeds maximum allowed limit of 50."
            raise ValueError(msg)

        if src_pot.is_symlink():
            target = os.path.realpath(str(src_pot.resolve(strict=True)))
            try:
                if os.path.commonpath([allowed_src_dir, target]) != allowed_src_dir:
                    msg = "Symlink target outside allowed directory"
                    raise ValueError(msg)
            except ValueError as e:
                msg = "Symlink target outside allowed directory"
                raise ValueError(msg) from e

        pot_dir.mkdir(parents=True, exist_ok=True)
        resolved_pot_dir = pot_dir.resolve()
        final_dest = resolved_pot_dir / f"generation_{iteration:03d}.yace"

        max_size = self.config.trainer.max_potential_size
        st = resolved_src.stat()
        if st.st_size > max_size:
            msg = f"Source potential file exceeds maximum allowed size ({max_size} bytes)"
            raise ValueError(msg)

        # Secure cross-filesystem atomic copy with streaming hash
        try:
            fd, tmp_path_str = tempfile.mkstemp(dir=str(resolved_pot_dir), prefix=".tmp_pot_")
            tmp_path = Path(tmp_path_str)
            try:
                sha256_src = hashlib.sha256()
                sha256_dest = hashlib.sha256()

                # We copy into the temporary file in the target directory and hash the source
                with os.fdopen(fd, "wb") as f_out, Path.open(resolved_src, "rb") as f_in:
                    for chunk in iter(lambda: f_in.read(8192), b""):
                        sha256_src.update(chunk)
                        f_out.write(chunk)

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

        _secure_resolve_and_validate_dir(str(tmp_work_dir), check_exists=False)
        _secure_resolve_and_validate_dir(str(work_dir), check_exists=False)
        import fcntl
        import shutil

        # To prevent race conditions and cross-filesystem issues, we will just use a direct copytree
        # onto a temporary resolved destination, then perform an atomic rename.
        final_dest = work_dir.resolve(strict=False)

        lock_name = getattr(self.config.loop_strategy, "swap_lock_file", ".swap.lock")
        lock_file = final_dest.parent / lock_name
        with Path.open(lock_file, "w") as f_lock:
            try:
                fcntl.flock(f_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as e:
                msg = f"Directory swap is currently locked by another process: {e}"
                raise RuntimeError(msg) from e

            temp_swap = Path(tempfile.mkdtemp(dir=str(final_dest.parent)))
            try:
                # Secure copy into the isolated temp dir
                shutil.copytree(str(tmp_work_dir), str(temp_swap / "new_work"))

                # Atomic swap
                if final_dest.exists():
                    backup_dest = temp_swap / "backup"
                    Path(final_dest).rename(backup_dest)
                    try:
                        Path(temp_swap / "new_work").rename(final_dest)
                    except Exception:
                        # Rollback
                        if final_dest.exists():
                            Path(backup_dest).replace(final_dest)
                        else:
                            Path(backup_dest).rename(final_dest)
                        raise
                else:
                    Path(temp_swap / "new_work").rename(final_dest)
            finally:
                shutil.rmtree(str(temp_swap), ignore_errors=True)
                # Clean up the original tmp_work_dir
                shutil.rmtree(str(tmp_work_dir), ignore_errors=True)
                fcntl.flock(f_lock, fcntl.LOCK_UN)

        # Best effort remove lock file
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

        # Validating interface target input
        if not hasattr(self.config.structure_generator, "valid_interface_targets"):
            msg = "Security Validation: valid_interface_targets list missing in configuration."
            raise ValueError(msg)

        allowed_targets = self.config.structure_generator.valid_interface_targets
        t = self.config.system.interface_target
        # Using string representation mapping for checking validation list: "element1/element2" format
        target_str = f"{t.element1}{t.element2}" if hasattr(t, "element1") else str(t)

        if target_str not in allowed_targets:
            msg = f"Security Violation: Interface target '{target_str}' is not in the trusted whitelist: {allowed_targets}"
            raise ValueError(msg)

        if len(target_str) > 100 or not re.match(r"^[a-zA-Z0-9]+$", target_str):
            msg = f"Security Violation: Invalid characters or size in interface target string: {target_str}"
            raise ValueError(msg)

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

        # Authorized base path for cleanups
        active_learning_dir_str = str(
            (self.config.project_root / "active_learning").resolve(strict=False)
        )
        allowed_extensions = getattr(
            self.config.loop_strategy,
            "allowed_cleanup_extensions",
            {".dat", ".wfc", ".lammps", ".yace"},
        )
        size_threshold = getattr(self.config.loop_strategy, "cleanup_size_threshold", 10240)

        for path in paths:
            try:
                if not path.exists() or not path.is_file():
                    continue

                canonical_path = path.resolve(strict=True)
                canonical_str = str(canonical_path)

                # Extension whitelist validation
                if (
                    canonical_path.suffix not in allowed_extensions
                    and canonical_path.name != "dump.lammps"
                ):
                    logging.warning(
                        f"Validation Error: Refusing to delete file with unauthorized extension: {canonical_str}"
                    )
                    continue

                # Ensure path is strictly within the allowed directory
                if not canonical_str.startswith(active_learning_dir_str):
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
                # Use Path.unlink() directly and catch exceptions instead of doing TOCTOU property checks
                # The st_uid check above is fine as a prerequisite, but we still rely on Path.unlink atomic success
                try:
                    canonical_path.unlink()
                    logging.info(f"Cleaned up large artifact securely: {canonical_str}")
                except OSError as e:
                    logging.warning(f"OS error during file deletion for {canonical_str}: {e}")
            except Exception as e:
                # Swallow error to ensure idempotency and non-blocking behavior
                logging.warning(f"Failed to securely cleanup artifact {path}: {e}")

    def run_cycle(self) -> str | None:  # noqa: PLR0911
        """Runs the 4-phase Hierarchical Distillation Workflow infinitely or bounded."""
        import tempfile
        import time

        max_iters = getattr(self.config.loop_strategy, "max_iterations", 1000)
        if not isinstance(max_iters, int) or max_iters <= 0:
            max_iters = 1000

        timeout_seconds = getattr(self.config.loop_strategy, "timeout_seconds", 86400)
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            timeout_seconds = 86400

        start_time = time.time()

        while self.iteration < max_iters:
            if time.time() - start_time > timeout_seconds:
                logging.error(
                    f"Global Orchestrator timeout reached ({timeout_seconds}s). Halting execution to prevent infinite loop."
                )
                return "TIMEOUT"

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
                                "dump_file": str(tmp_work_dir / "md_run" / "dump.lammps"),
                            }

                            # 2. Extract Intelligent Cluster & DFT
                            candidate_generator = self._select_candidates(halt_dict)

                            new_pot_path = self._run_dft_and_train(
                                candidate_generator, tmp_work_dir, current_pot
                            )

                            if isinstance(new_pot_path, str):
                                return new_pot_path

                            # 3. Finetune explicitly triggering hierarchical trainers
                            if isinstance(self.oracle, TieredOracle) and hasattr(
                                self.oracle, "primary_oracle"
                            ):
                                mace_model = getattr(
                                    self.oracle.primary_oracle, "mace_model_path", None
                                )
                                if mace_model:
                                    import tempfile

                                    from ase.io import read

                                    # Get training dataset path to use for finetuning
                                    data_dir = self.config.project_root / "data"
                                    dataset_path = data_dir / "accumulated.extxyz"
                                    if dataset_path.exists():
                                        try:
                                            structures = read(str(dataset_path), index=":")
                                            if not isinstance(structures, list):
                                                structures = [structures]

                                            with tempfile.TemporaryDirectory(
                                                dir=str(tmp_work_dir)
                                            ) as mace_tmp:
                                                mace_out_path = Path(mace_tmp) / "finetuned.model"
                                                new_mace_model = (
                                                    self.finetune_manager.finetune_mace(
                                                        structures=structures,
                                                        model_path=str(mace_model),
                                                        output_path=mace_out_path.parent,
                                                    )
                                                )
                                                # Update primary oracle with the awakened MACE model
                                                self.oracle.primary_oracle.mace_model_path = str(new_mace_model)  # type: ignore[attr-defined]
                                        except Exception as e:
                                            logging.warning(f"Finetuning MACE failed: {e}")

                            if not self._validate_potential(new_pot_path):
                                return "VALIDATION_FAILED"

                            final_dest_str = str(self._deploy_potential(tmp_work_dir))
                            self._finalize_directories(tmp_work_dir, work_dir)

                            # 4. Explicit Update state to MD_RESUME
                            self.checkpoint.set_state("CURRENT_PHASE", "PHASE3_MD_RESUME")
                            current_state = "PHASE3_MD_RESUME"

                            # Cleanup Artifacts
                            heavy_files = [
                                tmp_work_dir / f
                                for f in ["wfc.dat", "dump.lammps", "wavefunctions.wfc"]
                            ]
                            self._cleanup_artifacts(heavy_files)

                if current_state == "PHASE3_MD_RESUME":
                    logging.info("Resuming MD Exploration")
                    # Smoothly resume the simulation
                    pot_to_resume = (
                        Path(final_dest_str)
                        if "final_dest_str" in locals()
                        else self.get_latest_potential()
                    )
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
