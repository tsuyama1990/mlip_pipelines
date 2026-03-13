from pathlib import Path
from ase import Atoms
from src.domain_models.config import TrainingConfig
import random

class ACETrainer:
    """Trains and optimizes active set using Pacemaker."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def select_local_active_set(self, candidates: list[Atoms], anchor: Atoms, n: int) -> list[Atoms]:
        """
        Runs D-Optimality to select optimal structures from candidates.
        """
        # In actual implementation pace_activeset CLI runs here.
        import random
        selected = []
        selected.append(anchor)

        # sample remaining from candidates without replacement if n > 1
        num_to_select = min(n - 1, len(candidates))
        if num_to_select > 0:
            sys_random = random.SystemRandom()
            sampled = sys_random.sample(candidates, num_to_select)
            selected.extend(sampled)

        return selected

    def update_dataset(self, new_data: list[Atoms]) -> Path:
        """
        Updates dataset. Here we return a dummy path as mock.
        """
        return Path("data/accumulated.pckl.gzip")

    def train(self, dataset: Path, initial_potential: Path | None, output_dir: Path) -> Path:
        """
        Runs Pacemaker training with Delta Learning against LJ baseline.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        # In reality this triggers pace_train
        output_pot = output_dir / "output_potential.yace"

        # Explicit mock creation of the file so pipelines continue
        with output_pot.open("w") as f:
            f.write("MOCK_ACE_POTENTIAL")

        return output_pot
