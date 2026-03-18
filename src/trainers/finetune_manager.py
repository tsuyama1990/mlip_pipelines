import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write

from src.domain_models.config import TrainerConfig
from src.trainers.binary_resolver import BinaryResolverMixin


class FinetuneManager(BinaryResolverMixin):
    """Manages PyTorch-based MACE foundation model fine-tuning."""

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def _validate_output_path(self, output_path: Path) -> Path:
        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(output_path), check_exists=False)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        resolved_out = output_path.resolve(strict=True)

        if hasattr(self.config, "project_root"):
            proj_root = Path(self.config.project_root).resolve(strict=True)
            tmp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            if not str(resolved_out).startswith(str(proj_root)) and not str(
                resolved_out
            ).startswith(str(tmp_root)):
                msg = (
                    f"output_path is outside the trusted base directory or temp dir: {resolved_out}"
                )
                raise ValueError(msg)

        if not os.access(resolved_out, os.W_OK):
            msg = f"output_path is not writable: {resolved_out}"
            raise PermissionError(msg)

        return resolved_out

    def finetune_mace(self, structures: list[Atoms], model_path: str, output_path: Path) -> Path:
        """Fine-tunes the MACE foundation model using the provided structures."""
        mace_train_bin = self._resolve_binary_path(self.config.mace_train_binary, "mace_run_train")

        resolved_out = self._validate_output_path(output_path)

        # Refactored to reduce complexity
        with tempfile.TemporaryDirectory(prefix="pyacemaker_mace_") as tmp:
            temp_dir = Path(tmp).resolve(strict=True)
            self._verify_tmp_dir(temp_dir)
            train_xyz = temp_dir / "train.extxyz"
            self._secure_write_xyz(train_xyz, structures)
            self._run_mace_subprocess(mace_train_bin, train_xyz, model_path, temp_dir)
            return self._extract_and_cleanup_mace_output(temp_dir, resolved_out)

    def _verify_tmp_dir(self, temp_dir: Path) -> None:
        tmp_root = Path(tempfile.gettempdir()).resolve(strict=True)
        if not temp_dir.is_relative_to(tmp_root):
            msg = f"Temporary directory {temp_dir} is not within trusted temp root {tmp_root}."
            raise ValueError(msg)

    def _secure_write_xyz(self, train_xyz: Path, structures: list) -> None:
        """Secure atomic write."""
        import fcntl
        import os
        import tempfile

        if train_xyz.exists():
            msg = f"File already exists: {train_xyz}"
            raise FileExistsError(msg)

        # Write to a temporary file first in the same directory, then rename
        fd, tmp_path_str = tempfile.mkstemp(dir=str(train_xyz.parent), prefix=".tmp_finetune_")
        tmp_path = Path(tmp_path_str)
        try:
            # File locking using fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with os.fdopen(fd, "w", encoding="utf-8") as f_out:
                for atoms in structures:
                    write(f_out, atoms, format="extxyz")

            # Atomic rename (POSIX only, but fine for HPC)
            Path(tmp_path_str).replace(str(train_xyz))
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            msg = f"Failed to securely write xyz: {e}"
            raise RuntimeError(msg) from e

    def _run_mace_subprocess(
        self, mace_train_bin: str, train_xyz: Path, model_path: str, temp_dir: Path
    ) -> None:
        cmd = [
            mace_train_bin,
            "--train_file",
            str(train_xyz),
            "--model",
            model_path,
            "--output_dir",
            str(temp_dir),
        ]
        if self.config.mace_freeze_body:
            cmd.append("--freeze_body")
        cmd.extend(
            [
                "--max_num_epochs",
                str(self.config.mace_finetuning_epochs),
                "--lr",
                str(self.config.mace_learning_rate),
            ]
        )

        try:
            subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
                timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired as e:
            logging.exception(
                f"mace_run_train execution timed out after {self.config.timeout} seconds."
            )
            msg = "mace_run_train execution timed out."
            raise RuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            msg = f"mace_run_train execution failed: {e.stderr}"
            raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            msg = "mace_run_train executable not found in PATH."
            raise RuntimeError(msg) from e

    def _extract_and_cleanup_mace_output(self, temp_dir: Path, resolved_out: Path) -> Path:
        model_files = list(temp_dir.glob("*.model"))
        if not model_files:
            msg = "mace_run_train completed but failed to produce a .model file."
            raise RuntimeError(msg)

        shutil.copy2(str(model_files[0]), str(resolved_out / "finetuned.model"))

        try:
            shutil.rmtree(str(temp_dir), ignore_errors=False)
        except PermissionError as e:
            logging.warning(f"Could not fully remove temp directory {temp_dir}: {e}")
        except Exception as e:
            logging.warning(f"Error removing temp directory {temp_dir}: {e}")

        return resolved_out / "finetuned.model"
