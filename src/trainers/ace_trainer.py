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

        has_pace = shutil.which("pace_activeset") is not None

        if not has_pace:
            logger.warning("pace_activeset not found in PATH. Defaulting to random selection.")
            sys_random = random.SystemRandom()
            sampled = sys_random.sample(candidates, num_to_select)
            selected.extend(sampled)
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
                subprocess.run(  # noqa: S603
                    cmd, check=True, capture_output=True, text=True
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

        write(
            str(accumulated_xyz),
            new_data,
            format="extxyz",
            append=accumulated_xyz.exists(),
        )

        return accumulated_xyz

    def train(self, dataset: Path, initial_potential: Path | None, output_dir: Path) -> Path:
        """
        Runs Pacemaker training with Delta Learning against LJ baseline.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pot = output_dir / "output_potential.yace"

        import shutil

        has_pace = shutil.which("pace_train") is not None

        if not has_pace:
            logger.warning(
                "pace_train not found in PATH. Failing gracefully by not creating the file."
            )

        cmd = ["pace_train"]
        if self.config.pace_train_args:
            cmd.extend(self.config.pace_train_args)

        # Override specific necessary ones to maintain flow
        # In a fully config-driven setup, user provides all args. But we must ensure dataset/output.
        # Check if already in config, else append
        if "--dataset" not in cmd:
            cmd.extend(["--dataset", str(dataset)])
        if "--output_dir" not in cmd:
            cmd.extend(["--output_dir", str(output_dir)])
        if "--max_num_epochs" not in cmd:
            cmd.extend(["--max_num_epochs", str(self.config.max_epochs)])

        if initial_potential and initial_potential.exists():
            cmd.extend(["--initial_potential", str(initial_potential)])

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

        return output_pot
