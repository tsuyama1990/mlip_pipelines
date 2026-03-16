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
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        resolved_out = output_path.resolve(strict=True)

        if hasattr(self.config, "project_root"):
            proj_root = Path(self.config.project_root).resolve(strict=True)
            tmp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            if not str(resolved_out).startswith(str(proj_root)) and not str(resolved_out).startswith(str(tmp_root)):
                msg = f"output_path is outside the trusted base directory or temp dir: {resolved_out}"
                raise ValueError(msg)

        if not os.access(resolved_out, os.W_OK):
            msg = f"output_path is not writable: {resolved_out}"
            raise PermissionError(msg)

        return resolved_out

    def finetune_mace(
        self, structures: list[Atoms], model_path: str, output_path: Path
    ) -> Path:
        """Fine-tunes the MACE foundation model using the provided structures."""
        mace_train_bin = self._resolve_binary_path(
            self.config.mace_train_binary, "mace_run_train"
        )

        resolved_out = self._validate_output_path(output_path)

        temp_dir = Path(tempfile.mkdtemp(prefix="pyacemaker_mace_"))
        try:
            train_xyz = temp_dir / "train.extxyz"
            write(str(train_xyz), structures, format="extxyz")

            cmd = [
                mace_train_bin,
                "--train_file",
                str(train_xyz),
                "--model",
                model_path,
                "--output_dir",
                str(temp_dir),
                "--freeze_body",
                "--max_num_epochs",
                str(self.config.mace_finetuning_epochs),
                "--lr",
                str(self.config.mace_learning_rate),
            ]

            try:
                _res: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
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

            # Typically MACE saves the model with a .model extension
            # For simplicity, we assume we find one .model file or copy the entire output
            # In mock environment we'll assume it outputs something or we just create it.
            # UAT requirements: UAT-C05-03: return dummy model file
            model_files = list(temp_dir.glob("*.model"))
            if not model_files:
                msg = "mace_run_train completed but failed to produce a .model file."
                raise RuntimeError(msg)

            shutil.copy2(str(model_files[0]), str(resolved_out / "finetuned.model"))

            return resolved_out / "finetuned.model"

        finally:
            # UAT-C05-05 requires securely deleting and catching PermissionError
            try:
                shutil.rmtree(temp_dir, ignore_errors=False)
            except PermissionError as e:
                logging.warning(f"Could not fully remove temp directory {temp_dir}: {e}")
            except Exception as e:
                logging.warning(f"Error removing temp directory {temp_dir}: {e}")
