import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write

from src.core import AbstractTrainer
from src.domain_models.config import TrainerConfig


class PacemakerWrapper(AbstractTrainer):
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

    def _get_trusted_dirs(self) -> list[str]:
        trusted = list(self.config.trusted_directories)
        trusted.append(str(Path(sys.prefix) / "bin"))
        if hasattr(self.config, "project_root"):
            trusted.append(str(Path(self.config.project_root) / "bin"))
        return trusted

    def _validate_binary_properties(
        self, resolved_bin: Path, binary_name: str, trusted_dirs: list[str]
    ) -> None:
        if not resolved_bin.is_file() or not os.access(resolved_bin, os.X_OK):
            msg = f"Binary is not an executable file: {resolved_bin}"
            raise ValueError(msg)
        if resolved_bin.name != binary_name:
            msg = f"Resolved binary name must be '{binary_name}', got '{resolved_bin.name}'"
            raise ValueError(msg)
        is_trusted = False
        for trusted_path in trusted_dirs:
            try:
                resolved_trusted = Path(trusted_path).resolve(strict=True)
                if resolved_bin.is_relative_to(resolved_trusted):
                    is_trusted = True
                    break
            except OSError:
                continue
        if not is_trusted:
            msg = f"Resolved binary must reside in a trusted directory: {resolved_bin}"
            raise ValueError(msg)

    def _resolve_absolute_binary(
        self, binary_setting: str, binary_name: str, trusted_dirs: list[str]
    ) -> str:
        if ".." in binary_setting:
            msg = f"Invalid absolute binary path: {binary_setting}"
            raise ValueError(msg)
        resolved_bin = Path(os.path.realpath(binary_setting)).resolve(strict=True)
        self._validate_binary_properties(resolved_bin, binary_name, trusted_dirs)
        self._verify_hash(resolved_bin, binary_name)
        return str(resolved_bin)

    def _resolve_relative_binary(
        self, binary_setting: str, binary_name: str, trusted_dirs: list[str]
    ) -> str:
        if not re.match(r"^[-a-zA-Z0-9_.]+$", binary_setting):
            msg = f"Invalid binary name: {binary_setting}"
            raise ValueError(msg)
        resolved_which = shutil.which(binary_setting)
        if resolved_which is None:
            return binary_setting
        resolved_bin = Path(os.path.realpath(resolved_which)).resolve(strict=True)
        self._validate_binary_properties(resolved_bin, binary_name, trusted_dirs)
        self._verify_hash(resolved_bin, binary_name)
        return str(resolved_bin)

    def _verify_hash(self, resolved_bin: Path, binary_name: str) -> None:
        expected_hash = self.config.binary_hashes.get(binary_name)
        if expected_hash:
            import hashlib
            h = hashlib.sha256()
            with Path.open(resolved_bin, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                msg = f"Executable hash mismatch for {resolved_bin}"
                raise ValueError(msg)

    def _resolve_binary_path(self, binary_setting: str, binary_name: str) -> str:
        trusted_dirs = self._get_trusted_dirs()
        if Path(binary_setting).is_absolute():
            return self._resolve_absolute_binary(binary_setting, binary_name, trusted_dirs)
        return self._resolve_relative_binary(binary_setting, binary_name, trusted_dirs)

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
                "--input", in_file_str,
                "--output", out_file_str,
                "--n", str(n)
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

    def _validate_train_directories(self, dataset: Path, output_dir: Path) -> tuple[Path, Path]:
        resolved_dataset = dataset.resolve(strict=True)
        if not resolved_dataset.exists():
            msg = f"Dataset not found: {resolved_dataset}"
            raise FileNotFoundError(msg)

        # Verify it's an extxyz file
        if resolved_dataset.suffix != ".extxyz":
            msg = f"Dataset must be an .extxyz file, got: {resolved_dataset.name}"
            raise ValueError(msg)

        with Path.open(resolved_dataset, "r") as f:
            first_line = f.readline().strip()
            if not first_line.isdigit():
                msg = "Dataset does not appear to be a valid XYZ format (first line must be atom count)."
                raise ValueError(msg)

        # Force directory existence to allow strict resolution
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        resolved_output_dir = Path(output_dir).resolve(strict=True)

        if hasattr(self.config, "project_root"):
            proj_root = Path(self.config.project_root).resolve(strict=True)
            tmp_root = Path(tempfile.gettempdir()).resolve(strict=True)
            if not resolved_output_dir.is_relative_to(
                proj_root
            ) and not resolved_output_dir.is_relative_to(tmp_root):
                msg = f"output_dir is outside the trusted base directory or temp dir: {resolved_output_dir}"
                raise ValueError(msg)

        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(resolved_output_dir, os.W_OK):
            msg = f"output_dir is not writable: {resolved_output_dir}"
            raise PermissionError(msg)

        return resolved_dataset, resolved_output_dir

    def _build_train_command(
        self, pace_train_bin: str, dataset: Path, output_dir: Path, initial_potential: Path | None
    ) -> list[str]:
        if not re.match(r"^[-a-zA-Z0-9_.]+$", self.config.baseline_potential):
            msg = "Invalid baseline potential format"
            raise ValueError(msg)
        if not re.match(r"^[-a-zA-Z0-9_.]+$", self.config.regularization):
            msg = "Invalid regularization format"
            raise ValueError(msg)

        dataset_str = str(dataset.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", dataset_str) or ".." in dataset_str:
            msg = f"Invalid dataset path: {dataset_str}"
            raise ValueError(msg)

        output_dir_str = str(output_dir.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", output_dir_str) or ".." in output_dir_str:
            msg = f"Invalid output directory path: {output_dir_str}"
            raise ValueError(msg)

        cmd = [
            pace_train_bin,
            "--dataset", dataset_str,
            "--max_num_epochs", str(int(self.config.max_epochs)),
            "--active_set_size", str(int(self.config.active_set_size)),
            "--baseline_potential", self.config.baseline_potential,
            "--regularization", self.config.regularization,
            "--output_dir", output_dir_str
        ]

        if initial_potential and initial_potential.exists():
            resolved_init_pot = str(initial_potential.resolve(strict=True))
            if not re.match(r"^[/a-zA-Z0-9_.-]+$", resolved_init_pot) or ".." in resolved_init_pot:
                msg = f"Initial potential path contains invalid characters: {resolved_init_pot}"
                raise ValueError(msg)
            cmd.extend(["--initial_potential", resolved_init_pot])

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
                cmd, check=True, capture_output=True, text=True, shell=False
            )
        except subprocess.CalledProcessError as e:
            msg = f"pace_train execution failed: {e.stderr}"
            raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            msg = "pace_train executable not found in PATH."
            raise RuntimeError(msg) from e

        return out_pot
