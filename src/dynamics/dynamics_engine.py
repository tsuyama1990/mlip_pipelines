from pathlib import Path

from ase.build import bulk

from src.core.interfaces import AbstractDynamics
from src.domain_models.config import DynamicsConfig
from src.domain_models.dtos import ExplorationStrategy, HaltEvent


class DynamicsEngine(AbstractDynamics):
    def __init__(self, config: DynamicsConfig) -> None:
        self.config = config

    def run_exploration(
        self, potential_path: Path, strategy: ExplorationStrategy
    ) -> HaltEvent | None:
        """Simulate an MD run that might hit high uncertainty."""
        # In a real engine, we'd run lammps. For cycle 1, to prove it runs,
        # we check the strategy temperature. If it's too high, we throw a HaltEvent

        if strategy.t_schedule and max(strategy.t_schedule) > self.config.max_temperature_threshold:
            # We construct a dummy atoms object representing the state when halted
            halt_atoms = bulk("Fe", "bcc", a=2.8)

            return HaltEvent(
                timestep=50,
                max_gamma=self.config.gamma_threshold + 1.0,
                triggering_atoms_indices=[0],
                halt_structure=halt_atoms,
            )

        # Returning None implies the MD ran converged without hitting threshold
        return None
