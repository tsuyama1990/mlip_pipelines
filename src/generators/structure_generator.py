from typing import Any

from ase.build import bulk

from src.core.interfaces import AbstractGenerator
from src.domain_models.config import SystemConfig
from src.domain_models.dtos import ExplorationStrategy, HaltEvent


class StructureGenerator(AbstractGenerator):
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def generate_initial_structures(self, strategy: ExplorationStrategy) -> list[Any]:
        """Generate a basic bulk structure of the first element."""
        element = self.config.elements[0]
        structure = "fcc"
        if self.config.structure_type and element in self.config.structure_type:
            structure = self.config.structure_type[element]

        a = 3.6
        if self.config.lattice_parameters and element in self.config.lattice_parameters:
            a = self.config.lattice_parameters[element]

        atoms = bulk(element, structure, a=a)
        return [atoms]

    def generate_local_candidates(
        self, halt_event: HaltEvent, strategy: ExplorationStrategy
    ) -> list[Any]:
        """Apply random jitter to the halt structure to simulate normal mode sampling."""
        base_atoms = halt_event.halt_structure.copy()
        candidates = []
        for _ in range(strategy.n_defects):
            jittered = base_atoms.copy()
            jittered.rattle(stdev=strategy.strain_range)
            candidates.append(jittered)
        return candidates
