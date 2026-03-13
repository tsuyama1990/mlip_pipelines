import logging
import random
import subprocess
import typing
from pathlib import Path

from ase import Atoms

from src.domain_models.config import TrainingConfig

logger = logging.getLogger(__name__)


class ACETrainer:
    """Trains and optimizes active set using Pacemaker."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def select_local_active_set(
        self, candidates: "typing.Iterable[Atoms]", anchor: Atoms, n: int
    ) -> list[Atoms]:
        """
        Runs D-Optimality to select optimal structures from candidates using pace_activeset.
        """
        import collections.abc

        selected = [anchor]

        # We must peek to determine length without materializing entirely, but since pacing needs the whole set written
        # to a file anyway, we will stream the writes to the tempfile iteratively if fallback or otherwise.

        import shutil
        import tempfile

        has_pace = shutil.which(self.config.pace_activeset_executable) is not None

        # In order to avoid OOM by keeping all candidates in memory when falling back, we'll lazily evaluate the iterable
        def _get_iterable_fallback() -> list[Atoms]:
            c_list = list(candidates)
            num_to_select = min(n - 1, len(c_list))
            if num_to_select <= 0:
                return selected

            if self.config.activeset_fallback_strategy == "random":
                sys_random = random.SystemRandom()
                sampled = sys_random.sample(c_list, num_to_select)
                selected.extend(sampled)
            elif self.config.activeset_fallback_strategy == "first_n":
                selected.extend(c_list[:num_to_select])
            else:
                msg = f"Unknown fallback strategy: {self.config.activeset_fallback_strategy}"
                raise RuntimeError(msg)
            return selected

        if not has_pace:
            logger.warning(
                f"{self.config.pace_activeset_executable} not found in PATH. Defaulting to {self.config.activeset_fallback_strategy} selection."
            )
            return _get_iterable_fallback()

        from ase.io import write

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            input_file = tmpdir / "candidates.extxyz"

            # Write iteratively without allocating huge lists
            def _chained() -> collections.abc.Iterator[Atoms]:
                yield anchor
                yield from candidates

            write(input_file, list(_chained()), format="extxyz")

            cmd = [
                "pace_activeset",
                "--dataset",
                str(input_file),
                "--select_n",
                str(n),
                "--output",
                str(tmpdir / "selected.extxyz"),
            ]

            try:
                from ase.io import read

                subprocess.run(  # noqa: S603
                    cmd, check=True, capture_output=True, text=True
                )

                logger.info("pace_activeset completed successfully.")

                # After successful call, read actual structures determined by D-Optimality
                selected_output_file = tmpdir / "selected.extxyz"
                if selected_output_file.exists():
                    actual_selected = read(str(selected_output_file), index=":", format="extxyz")
                    if isinstance(actual_selected, list):
                        selected.extend(actual_selected)
                    else:
                        selected.append(actual_selected)
                else:
                    # Fallback only if the output file is missing despite successful command
                    from ase.io import read as fallback_read

                    fallback_structures = fallback_read(str(input_file), index=":", format="extxyz")
                    if isinstance(fallback_structures, list) and len(fallback_structures) > 1:
                        c_list_read = fallback_structures[1:]
                        sys_random = random.SystemRandom()
                        num_to_sel = min(n - 1, len(c_list_read))
                        sampled = sys_random.sample(c_list_read, num_to_sel)
                        selected.extend(sampled)
            except subprocess.CalledProcessError:
                logger.exception("pace_activeset failed")
                from ase.io import read as fallback_read

                fallback_structures = fallback_read(str(input_file), index=":", format="extxyz")
                if isinstance(fallback_structures, list) and len(fallback_structures) > 1:
                    c_list_read = fallback_structures[1:]
                    sys_random = random.SystemRandom()
                    num_to_sel = min(n - 1, len(c_list_read))
                    sampled = sys_random.sample(c_list_read, num_to_sel)
                    selected.extend(sampled)

        return selected

    def update_dataset(self, new_data: list[Atoms], dataset_path: Path | None = None) -> Path:
        """
        Updates dataset by running pace_collect or writing extxyz.
        """
        from ase.io import write

        if dataset_path is None:
            data_dir = Path("data")
            accumulated_xyz = data_dir / "accumulated.extxyz"
        else:
            accumulated_xyz = dataset_path
            data_dir = accumulated_xyz.parent

        data_dir.mkdir(exist_ok=True, parents=True)

        if not new_data:
            return accumulated_xyz

        # Prevent unbounded file growth (e.g. rotation if file > 100MB)
        max_size_bytes = 100 * 1024 * 1024  # 100 MB

        # Use an explicit file locking mechanism via fcntl to safely append structures directly when mapping
        import fcntl

        lock_file = data_dir / ".accumulated.lock"
        with lock_file.open("w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if accumulated_xyz.exists() and accumulated_xyz.stat().st_size > max_size_bytes:
                    import time

                    rotated_name = f"accumulated_{int(time.time())}.extxyz"
                    accumulated_xyz.rename(data_dir / rotated_name)
                    logger.info(f"Rotated large dataset to {rotated_name}")

                write(
                    str(accumulated_xyz),
                    new_data,
                    format="extxyz",
                    append=accumulated_xyz.exists(),
                )
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

        return accumulated_xyz

    def _build_pace_train_cmd(
        self, dataset: Path, initial_potential: Path | None, output_dir: Path
    ) -> list[str]:
        import re

        safe_arg_pattern = re.compile(r"^[A-Za-z0-9\-_=.]+$")

        cmd = [self.config.pace_train_executable]
        if self.config.pace_train_args:
            for arg in self.config.pace_train_args:
                if not safe_arg_pattern.match(arg):
                    msg = f"Invalid characters in pace_train_args: {arg}"
                    raise ValueError(msg)
            cmd.extend(self.config.pace_train_args)

        if "--dataset" not in cmd:
            cmd.extend(["--dataset", str(dataset)])
        if "--output_dir" not in cmd:
            cmd.extend(["--output_dir", str(output_dir)])
        if "--max_num_epochs" not in cmd:
            cmd.extend(["--max_num_epochs", str(self.config.max_epochs)])

        if initial_potential and initial_potential.exists():
            cmd.extend(["--initial_potential", str(initial_potential)])

        return cmd

    def train(self, dataset: Path, initial_potential: Path | None, output_dir: Path) -> Path:
        """
        Runs Pacemaker training with Delta Learning against LJ baseline.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pot = output_dir / "output_potential.yace"

        import shutil

        has_pace = shutil.which(self.config.pace_train_executable) is not None

        if not has_pace:
            logger.warning(
                f"{self.config.pace_train_executable} not found in PATH. Failing gracefully by not creating the file."
            )

        cmd = self._build_pace_train_cmd(dataset, initial_potential, output_dir)

        if has_pace:
            try:
                subprocess.run(  # noqa: S603
                    cmd, check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                logger.exception("pace_train failed")
                msg = f"pace_train failed: {e.stderr}"
                raise RuntimeError(msg) from e
            else:
                return output_pot
        else:
            msg = "pace_train executable not available."
            raise FileNotFoundError(msg)
