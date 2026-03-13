from ase import Atoms

from src.domain_models.config import StructureGeneratorConfig


class StructureGenerator:
    """Generates localized candidate structures around an uncertain anchor."""

    def __init__(self, config: StructureGeneratorConfig) -> None:
        self.config = config

    def generate_local_candidates(self, s0: Atoms, n: int = 20) -> list[Atoms]:
        """Generates candidates via random rattling using streaming generation."""
        from collections.abc import Iterator

        # Scale down n if the structure is massive to avoid OOM
        actual_n = n if len(s0) < 1000 else max(1, n // 10)

        def _generator() -> Iterator[Atoms]:
            for i in range(actual_n):
                c = s0.copy()  # type: ignore[no-untyped-call]
                c.rattle(stdev=self.config.stdev, seed=self.config.seed_base + i)
                yield c

        # In a fully streaming application this would yield directly,
        # but the trainer pipeline expects a list for selection sampling.
        # This wrapper explicitly bounds the generation safely.
        return list(_generator())
