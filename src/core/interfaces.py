from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.domain_models.dtos import ExplorationStrategy, HaltEvent, ValidationScore


class AbstractGenerator(ABC):
    @abstractmethod
    def generate_initial_structures(self, strategy: ExplorationStrategy) -> list[Any]:
        """Generate initial structures based on exploration strategy."""

    @abstractmethod
    def generate_local_candidates(self, halt_event: HaltEvent, strategy: ExplorationStrategy) -> list[Any]:
        """Generate local candidate structures around a halted state."""

class AbstractOracle(ABC):
    @abstractmethod
    def compute(self, structures: list[Any]) -> list[Any]:
        """Compute exact properties (forces, energies) for the given structures."""

class AbstractTrainer(ABC):
    @abstractmethod
    def train(self, structures: list[Any]) -> Path:
        """Train the potential using the provided structures. Returns path to new potential."""

    @abstractmethod
    def filter_active_set(self, candidates: list[Any], anchor: Any) -> list[Any]:
        """Filter structures using active set logic (e.g. D-Optimality)."""

class AbstractDynamics(ABC):
    @abstractmethod
    def run_exploration(self, potential_path: Path, strategy: ExplorationStrategy) -> HaltEvent | None:
        """Run dynamics and return a HaltEvent if interrupted, otherwise None."""

class AbstractValidator(ABC):
    @abstractmethod
    def validate(self, potential_path: Path) -> ValidationScore:
        """Validate the potential and return a ValidationScore."""
