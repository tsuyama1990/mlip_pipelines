from ase import Atoms

from src.domain_models.config import StructureGeneratorConfig


class StructureGenerator:
    """Generates localized candidate structures around an uncertain anchor."""

    def __init__(self, config: StructureGeneratorConfig) -> None:
        self.config = config

    def generate_local_candidates(self, s0: Atoms, n: int = 20) -> list[Atoms]:
        """Generates candidates via random rattling."""
        candidates = []
        for i in range(n):
            c = s0.copy()  # type: ignore[no-untyped-call]
            c.rattle(stdev=self.config.stdev, seed=self.config.seed_base + i)
            candidates.append(c)
        return candidates
