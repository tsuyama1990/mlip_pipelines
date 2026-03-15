import abc
import typing
from pathlib import Path
from typing import Any

from ase import Atoms

from src.core.exceptions import DynamicsHaltInterrupt, OracleConvergenceError

__all__ = [
    "AbstractDynamics",
    "AbstractGenerator",
    "AbstractOracle",
    "AbstractTrainer",
    "DynamicsHaltInterrupt",
    "OracleConvergenceError",
]


class AbstractOracle(abc.ABC):
    """Abstract Base Class for the DFT Oracle module."""

    @abc.abstractmethod
    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """Runs DFT computation on a batch of structures."""


class AbstractTrainer(abc.ABC):
    """Abstract Base Class for the ACE Trainer module."""

    @abc.abstractmethod
    def update_dataset(self, new_atoms_list: list[Atoms], dataset_path: Path) -> Path:
        """Appends new structures to the dataset."""

    @abc.abstractmethod
    def train(self, dataset: Path, initial_potential: Path | None, output_dir: Path) -> Path:
        """Trains the potential on the accumulated dataset."""

    @abc.abstractmethod
    def select_local_active_set(
        self, candidates: list[Atoms], anchor: Atoms, n: int = 5
    ) -> list[Atoms]:
        """Selects the best structures using D-Optimality."""


class AbstractDynamics(abc.ABC):
    """Abstract Base Class for Dynamics (MD/KMC) Engines."""

    @abc.abstractmethod
    def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:
        """Runs MD or KMC exploration until a halt condition or completion."""


class AbstractGenerator(abc.ABC):
    """Abstract Base Class for Structure Generators."""

    @abc.abstractmethod
    def generate_local_candidates(self, s0: Atoms, n: int = 20) -> typing.Iterator[Atoms]:
        """Generates perturbed candidates around a structure."""
