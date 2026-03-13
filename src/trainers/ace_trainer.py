import json
from pathlib import Path
from typing import Any

from src.core.interfaces import AbstractTrainer
from src.domain_models.config import TrainerConfig


class ACETrainer(AbstractTrainer):
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def train(self, structures: list[Any]) -> Path:
        """Saves a 'model' string representing training to disk and returns path."""
        model_path = Path("potential.yace")

        # Simple string representation logic mimicking model construction
        atoms_summary = [len(a) for a in structures]
        metadata = {"trained_on": atoms_summary, "max_degree": self.config.ace_max_degree}

        with model_path.open("w") as f:
            json.dump(metadata, f)

        return model_path

    def filter_active_set(self, candidates: list[Any], anchor: Any) -> list[Any]:
        """Filters the active set, removing empty structures."""
        filtered = [c for c in candidates if len(c) > 0]
        # Anchor must be included
        if anchor not in filtered:
            filtered.append(anchor)
        return filtered
