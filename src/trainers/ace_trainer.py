import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write

from src.core import AbstractTrainer
from src.domain_models.config import TrainerConfig
from src.trainers.binary_resolver import BinaryResolverMixin


class PacemakerWrapper(AbstractTrainer, BinaryResolverMixin):
    """Manages Pacemaker active set selection and training."""

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def update_dataset(self, new_atoms_list: list[Atoms], dataset_path: Path) -> Path:
        """Appends new structures to the dataset using a single streaming operation."""
        resolved_path = dataset_path.resolve()

        if hasattr(self.config, "project_root"):
            proj_root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_path.is_relative_to(proj_root):
                msg = f"Dataset path is outside the trusted base directory: {resolved_path}"
                raise ValueError(msg)

        if not resolved_path.parent.exists():
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

        if not resolved_path.exists():
            fd, temp_path_str = tempfile.mkstemp(dir=str(resolved_path.parent), suffix=".extxyz")
            os.close(fd)
            temp_path = Path(temp_path_str)
            try:
                write(temp_path_str, new_atoms_list, format="extxyz")
                temp_path.replace(resolved_path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise
            return resolved_path

        fd, temp_path_str = tempfile.mkstemp(dir=str(resolved_path.parent), suffix=".extxyz")
        os.close(fd)
        temp_path = Path(temp_path_str)
        try:
            shutil.copy2(str(resolved_path), temp_path_str)
            write(temp_path_str, new_atoms_list, format="extxyz", append=True)
            temp_path.replace(resolved_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return resolved_path

    def select_local_active_set(
        self, candidates: list[Atoms], anchor: Atoms, n: int = 5
    ) -> list[Atoms]:
        """Local D-Optimality selection of candidates."""
        from ase.io import read

        if not isinstance(n, int) or n < 1:
            msg = "n must be a positive integer"
            raise ValueError(msg)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            all_atoms = [anchor, *candidates]
            in_file = td_path / "candidates.extxyz"
            out_file = td_path / "selected.extxyz"
            write(str(in_file), all_atoms, format="extxyz")

            pace_activeset_bin = self._resolve_binary_path(
                self.config.pace_activeset_binary, "pace_activeset"
            )

            in_file_str = str(in_file.resolve(strict=True))
            out_file_str = str(out_file.resolve(strict=False))

            if not re.match(r"^[/a-zA-Z0-9_.-]+$", in_file_str) or ".." in in_file_str:
                msg = f"Invalid input file path: {in_file_str}"
                raise ValueError(msg)
            if not re.match(r"^[/a-zA-Z0-9_.-]+$", out_file_str) or ".." in out_file_str:
                msg = f"Invalid output file path: {out_file_str}"
                raise ValueError(msg)

            cmd = [
                pace_activeset_bin,
                "--input",
                in_file_str,
                "--output",
                out_file_str,
                "--n",
                str(n),
            ]

            try:
                _res: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
                    cmd, check=True, capture_output=True, text=True, shell=False
                )
                if out_file.exists():
                    selected = read(str(out_file), index=":")
                    if not isinstance(selected, list):
                        selected = [selected]
                    return selected
            except subprocess.CalledProcessError as e:
                msg = f"pace_activeset failed: {e.stderr}"
                raise RuntimeError(msg) from e
            except FileNotFoundError as e:
                msg = "pace_activeset executable not found in PATH."
                raise RuntimeError(msg) from e

            msg = "pace_activeset did not generate the output file."
            raise RuntimeError(msg)

    def manage_replay_buffer(
        self, new_surrogate_data: list[Atoms], history_file_path: Path, buffer_size: int
    ) -> list[Atoms]:
        """Samples the replay buffer to prevent catastrophic forgetting and returns combined training set.
        Uses O(N) Reservoir Sampling with streaming interfaces to prevent OOM errors on massive datasets.
        """
        import random

        from ase.io import iread

        resolved_history = history_file_path.resolve()
        sampled_history: list[Atoms] = []

        if resolved_history.exists():
            try:
                # O(N) Reservoir Sampling to avoid loading all history into memory
                for i, atoms in enumerate(iread(str(resolved_history), format="extxyz")):
                    if i < buffer_size:
                        sampled_history.append(atoms)
                    else:
                        import secrets

                        j = secrets.randbelow(i + 1)
                        if j < buffer_size:
                            sampled_history[j] = atoms
            except Exception as e:
                import logging

                logging.warning(f"Failed to read history file {resolved_history}: {e}")

        # Combine and shuffle
        combined_data = sampled_history + new_surrogate_data
        random.shuffle(combined_data)

        # Persist new surrogate data to history securely
        self.update_dataset(new_surrogate_data, history_file_path)

        return combined_data

    def _validate_train_directories(self, dataset: Path, output_dir: Path) -> tuple[Path, Path]:
        import os
        import tempfile

        if not dataset.exists():
            msg = f"Dataset not found: {dataset}"
            raise FileNotFoundError(msg)

        # Secure atomic resolution
        resolved_dataset = dataset.resolve(strict=True)

        # Basic path traversal prevention checks on the original inputs before resolution
        if ".." in str(dataset) or ".." in str(output_dir):
            msg = "Path traversal characters detected"
            raise ValueError(msg)

        if not str(resolved_dataset.name).lower().endswith(".extxyz"):
            msg = f"Dataset must be an .extxyz file, got: {resolved_dataset.name}"
            raise ValueError(msg)

        # Atomic file validation

        st = resolved_dataset.stat()
        max_dataset_size = 10 * 1024 * 1024 * 1024  # 10 GB limit for dataset
        if st.st_size > max_dataset_size:
            msg = f"Dataset size exceeds maximum limit of 10 GB: {st.st_size} bytes"
            raise ValueError(msg)

        try:
            with Path.open(resolved_dataset, "r", encoding="utf-8") as f:
                first_line: str = f.readline().strip()
                if not first_line.isdigit():
                    msg = "Dataset does not appear to be a valid XYZ format (first line must be atom count)."
                    raise ValueError(msg)

                # Check for basic content injection / sanity
                if "\x00" in first_line:
                    msg = "Null bytes detected in dataset file"
                    raise ValueError(msg)
        except OSError as e:
            msg = f"Failed to read dataset: {e}"
            raise ValueError(msg) from e

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        resolved_output_dir = output_dir.resolve(strict=True)

        if hasattr(self.config, "project_root"):
            proj_root = Path(self.config.project_root).resolve(strict=True)
            tmp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            if not str(resolved_output_dir).startswith(str(proj_root)) and not str(
                resolved_output_dir
            ).startswith(str(tmp_root)):
                msg = f"output_dir is outside the trusted base directory or temp dir: {resolved_output_dir}"
                raise ValueError(msg)

        if not os.access(resolved_output_dir, os.W_OK):
            msg = f"output_dir is not writable: {resolved_output_dir}"
            raise PermissionError(msg)

        return resolved_dataset, resolved_output_dir

    def _build_train_command(
        self, pace_train_bin: str, dataset: Path, output_dir: Path, initial_potential: Path | None
    ) -> list[str]:
        # Path validation without using regex, relying instead on Path strict resolution
        dataset_str = str(dataset.resolve(strict=True))
        output_dir_str = str(output_dir.resolve(strict=True))

        # Validate configuration values natively rather than with strict regexes
        # Whitelist approaches for categorical parameters:
        allowed_baselines = ["lj", "zbl", "none"]
        if self.config.baseline_potential.lower() not in allowed_baselines and not re.match(
            r"^[a-zA-Z0-9_-]+$", self.config.baseline_potential
        ):
            msg = f"Invalid baseline potential format: {self.config.baseline_potential}"
            raise ValueError(msg)

        if not re.match(r"^[a-zA-Z0-9_-]+$", self.config.regularization):
            msg = "Invalid regularization format"
            raise ValueError(msg)

        if any(char in self.config.baseline_potential for char in [';', '&', '|', '$', '`']):
            msg = "Baseline potential format contains illegal shell characters"
            raise ValueError(msg)

        if any(char in self.config.regularization for char in [';', '&', '|', '$', '`']):
            msg = "Regularization format contains illegal shell characters"
            raise ValueError(msg)

        # Delta Learning explicitly enabled: scale down max_num_epochs if we have an initial_potential
        epochs = int(self.config.max_epochs)
        if initial_potential and initial_potential.exists():
            epochs = max(1, epochs // self.config.delta_learning_epoch_scaling)

        cmd = [
            pace_train_bin,
            "--dataset",
            dataset_str,
            "--max_num_epochs",
            str(epochs),
            "--active_set_size",
            str(int(self.config.active_set_size)),
            "--baseline_potential",
            self.config.baseline_potential,
            "--regularization",
            self.config.regularization,
            "--output_dir",
            output_dir_str,
        ]

        if initial_potential and initial_potential.exists():
            resolved_init_pot = initial_potential.resolve(strict=True)
            canon_init_pot = str(resolved_init_pot)

            # Use strict whitelist pattern matching for each path component
            for part in Path(canon_init_pot).parts:
                if not re.match(r"^[-a-zA-Z0-9_.]+$", part) and part != "/":
                    msg = f"Security Violation: Initial potential path contains invalid characters: {canon_init_pot}"
                    raise ValueError(msg)

            if len(Path(canon_init_pot).parts) > 50:
                msg = "Security Violation: Path depth exceeds limits"
                raise ValueError(msg)

            import tempfile

            tmp_root = str(Path(tempfile.gettempdir()).resolve(strict=False))

            if not canon_init_pot.startswith(str(Path.cwd())) and not canon_init_pot.startswith(
                tmp_root
            ):
                msg = "Initial potential outside working directory or temp directory"
                raise ValueError(msg)

            cmd.extend(["--initial_potential", canon_init_pot])

        return cmd

    def train(self, dataset: Path, initial_potential: Path | None, output_dir: Path) -> Path:
        """Trains or fine-tunes the ACE model."""
        resolved_dataset, resolved_output_dir = self._validate_train_directories(
            dataset, output_dir
        )
        out_pot = resolved_output_dir / "output_potential.yace"

        pace_train_bin = self._resolve_binary_path(self.config.pace_train_binary, "pace_train")
        cmd = self._build_train_command(
            pace_train_bin, resolved_dataset, resolved_output_dir, initial_potential
        )

        try:
            _res2: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
                timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired as e:
            import logging

            logging.exception(
                f"pace_train execution timed out after {self.config.timeout} seconds."
            )
            msg = "pace_train execution timed out."
            raise RuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            msg = f"pace_train execution failed: {e.stderr}"
            raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            msg = "pace_train executable not found in PATH."
            raise RuntimeError(msg) from e

        return out_pot
