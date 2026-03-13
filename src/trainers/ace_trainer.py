import subprocess
from pathlib import Path

from ase import Atoms
from ase.io import write

from src.domain_models.config import TrainerConfig


class PacemakerWrapper:
    """Manages Pacemaker active set selection and training."""

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def update_dataset(self, new_atoms_list: list[Atoms], dataset_path: Path) -> Path:
        """Appends new structures to the dataset using a single streaming operation."""
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Use ase.io.write with the list directly since ASE handles lists
        # in a single file-open block much more efficiently than looping with append=True
        # which opens/closes the file descriptor for each single atom.
        if not dataset_path.exists():
            write(str(dataset_path), new_atoms_list, format="extxyz")
        else:
            write(str(dataset_path), new_atoms_list, format="extxyz", append=True)
        return dataset_path

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

            cmd = [
                "pace_activeset",
                "--input",
                str(in_file),
                "--output",
                str(out_file),
                "--n",
                str(n),
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, shell=False)
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

        cmd = [
            "pace_train",
            "--dataset",
            str(dataset),
            "--max_num_epochs",
            str(self.config.max_epochs),
            "--active_set_size",
            str(self.config.active_set_size),
            "--baseline_potential",
            self.config.baseline_potential,
            "--regularization",
            self.config.regularization,
            "--output_dir",
            str(output_dir),
        ]
        if initial_potential and initial_potential.exists():
            cmd.extend(["--initial_potential", str(initial_potential)])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, shell=False)
        except subprocess.CalledProcessError as e:
            msg = f"pace_train execution failed: {e.stderr}"
            raise RuntimeError(msg) from e
        except FileNotFoundError as e:
            # Re-raise explicit error instead of mocking dummy to adhere to NO MOCKS rule
            msg = "pace_train executable not found in PATH."
            raise RuntimeError(msg) from e

        return out_pot
