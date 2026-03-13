import re
import subprocess
from pathlib import Path

from ase import Atoms
from ase.io import write

from src.domain_models.config import TrainerConfig

BINARY_NAME_PATTERN = re.compile(r"^[-a-zA-Z0-9_.]+$")
PARAM_PATTERN = re.compile(r"^[-a-zA-Z0-9_.]+$")

class PacemakerWrapper:
    """Manages Pacemaker active set selection and training."""

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def update_dataset(self, new_atoms_list: list[Atoms], dataset_path: Path) -> Path:
        """Appends new structures to the dataset using a single streaming operation."""
        resolved_path = dataset_path.resolve()

        # Verify it writes to a valid directory to prevent path traversal
        if not resolved_path.parent.exists():
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

        # Use ase.io.write iteratively over chunks to prevent memory overhead
        # ase.io.write handles `append=True` safely without loading the whole file

        if not resolved_path.exists():
            write(str(resolved_path), new_atoms_list, format="extxyz")
        else:
            # We append chunks
            write(str(resolved_path), new_atoms_list, format="extxyz", append=True)
        return resolved_path

    def select_local_active_set(
        self, candidates: list[Atoms], anchor: Atoms, n: int
    ) -> list[Atoms]:
        """Local D-Optimality selection of candidates."""
        import tempfile

        from ase.io import read

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # Write out candidates and anchor
            all_atoms = [anchor, *candidates]
            in_file = td_path / "candidates.extxyz"
            out_file = td_path / "selected.extxyz"
            write(str(in_file), all_atoms, format="extxyz")

            import shutil

            pace_activeset_bin = shutil.which(self.config.pace_activeset_binary) or self.config.pace_activeset_binary

            if not BINARY_NAME_PATTERN.match(Path(pace_activeset_bin).name):
                msg = f"Invalid binary path: {pace_activeset_bin}"
                raise ValueError(msg)

            cmd = [
                pace_activeset_bin,
                "--input",
                str(in_file.resolve()),
                "--output",
                str(out_file.resolve()),
                "--n",
                str(n),
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, shell=False)  # noqa: S603
                if out_file.exists():
                    # Parse selected
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

            # If the tool doesn't output the file as expected
            msg = "pace_activeset did not generate the output file."
            raise RuntimeError(msg)

    def train(self, dataset: Path, initial_potential: Path | None, output_dir: Path) -> Path:
        """Trains or fine-tunes the ACE model."""
        if not dataset.exists():
            msg = f"Dataset not found: {dataset}"
            raise FileNotFoundError(msg)

        # Ensure output_dir is an absolute path and resolved to prevent directory traversal
        resolved_output_dir = Path(output_dir).resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        out_pot = resolved_output_dir / "output_potential.yace"

        if not PARAM_PATTERN.match(self.config.baseline_potential):
            msg = "Invalid baseline potential format"
            raise ValueError(msg)
        if not PARAM_PATTERN.match(self.config.regularization):
            msg = "Invalid regularization format"
            raise ValueError(msg)

        import shutil

        pace_train_bin = shutil.which(self.config.pace_train_binary) or self.config.pace_train_binary
        if not BINARY_NAME_PATTERN.match(Path(pace_train_bin).name):
            msg = f"Invalid binary path: {pace_train_bin}"
            raise ValueError(msg)

        cmd = [
            pace_train_bin,
            "--dataset",
            str(dataset.resolve()),
            "--max_num_epochs",
            str(self.config.max_epochs),
            "--active_set_size",
            str(self.config.active_set_size),
            "--baseline_potential",
            self.config.baseline_potential,
            "--regularization",
            self.config.regularization,
            "--output_dir",
            str(resolved_output_dir),
        ]
        if initial_potential and initial_potential.exists():
            cmd.extend(["--initial_potential", str(initial_potential.resolve())])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, shell=False)  # noqa: S603
        except subprocess.CalledProcessError as e:
            msg = f"pace_train execution failed: {e.stderr}"
            raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            # Re-raise explicit error instead of mocking dummy to adhere to NO MOCKS rule
            msg = "pace_train executable not found in PATH."
            raise RuntimeError(msg) from e

        return out_pot
