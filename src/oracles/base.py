import abc
from pathlib import Path

from ase import Atoms


class BaseOracle(abc.ABC):
    mace_model_path: str | None = None
    """Abstract Base Class for all Oracles."""

    @abc.abstractmethod
    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """Runs computation on a batch of structures."""
