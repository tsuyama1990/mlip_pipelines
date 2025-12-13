from abc import ABC, abstractmethod
import numpy as np
from ase import Atoms

class AbstractPotential(ABC):
    @abstractmethod
    def train(self, training_data: list[Atoms]) -> None:
        """Fine-tune the model with new data."""
        pass

    @abstractmethod
    def predict(self, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Return (energy [eV], forces [eV/A], stress [eV/A^3]).
        Stress should be (3, 3) matrix or Voigt notation (6,).
        """
        pass

    @abstractmethod
    def get_uncertainty(self, atoms: Atoms) -> np.ndarray:
        """
        Return per-atom uncertainty scores.
        Shape: (n_atoms,)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

class AbstractOracle(ABC):
    @abstractmethod
    def compute(self, atoms: Atoms) -> Atoms:
        """
        Perform DFT calculation.
        Returns atoms with calculated energy/forces attached.
        Raises OracleComputationError if calculation fails.
        """
        pass
