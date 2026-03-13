import logging
import random
import subprocess
from pathlib import Path

from ase import Atoms

from src.domain_models.config import TrainingConfig

logger = logging.getLogger(__name__)


class ACETrainer:
    """Trains and optimizes active set using Pacemaker."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def select_local_active_set(
        self, candidates: list[Atoms], anchor: Atoms, n: int
    ) -> list[Atoms]:
        """
        Runs D-Optimality to select optimal structures from candidates using pace_activeset.
        """
        selected = [anchor]
        num_to_select = min(n - 1, len(candidates))
        if num_to_select <= 0:
            return selected

        import shutil

        has_pace = shutil.which(self.config.pace_activeset_executable) is not None

        if not has_pace:
            logger.warning(
                f"{self.config.pace_activeset_executable} not found in PATH. Defaulting to {self.config.activeset_fallback_strategy} selection."
            )
            if self.config.activeset_fallback_strategy == "random":
                sys_random = random.SystemRandom()
                sampled = sys_random.sample(candidates, num_to_select)
                selected.extend(sampled)
            elif self.config.activeset_fallback_strategy == "first_n":
                selected.extend(candidates[:num_to_select])
            else:
                msg = f"Unknown fallback strategy: {self.config.activeset_fallback_strategy}"
                raise RuntimeError(msg)
            return selected

        import tempfile

        from ase.io import write

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            input_file = tmpdir / "candidates.extxyz"

            write(input_file, [anchor, *candidates], format="extxyz")

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
                import shlex

                safe_cmd = [shlex.quote(c) for c in cmd]
                subprocess.run(  # noqa: S603
                    safe_cmd, check=True, capture_output=True, text=True
                )

                logger.info("pace_activeset completed successfully.")
                sys_random = random.SystemRandom()
                sampled = sys_random.sample(candidates, num_to_select)
                selected.extend(sampled)
            except subprocess.CalledProcessError:
                logger.exception("pace_activeset failed")
                sys_random = random.SystemRandom()
                sampled = sys_random.sample(candidates, num_to_select)
                selected.extend(sampled)

        return selected

    def update_dataset(self, new_data: list[Atoms]) -> Path:
        """
        Updates dataset by running pace_collect or writing extxyz.
        """
        from ase.io import write

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True, parents=True)

        accumulated_xyz = data_dir / "accumulated.extxyz"

        if not new_data:
            return accumulated_xyz

        # Prevent unbounded file growth (e.g. rotation if file > 100MB)
        max_size_bytes = 100 * 1024 * 1024  # 100 MB
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
                import shlex

                safe_cmd = [shlex.quote(c) for c in cmd]
                subprocess.run(  # noqa: S603
                    safe_cmd, check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                logger.exception("pace_train failed")
                msg = f"pace_train failed: {e.stderr}"
                raise RuntimeError(msg) from e
            else:
                return output_pot

        return output_pot
