from ase import Atoms


class StructureGenerator:
    """Generates localized candidate structures around an uncertain anchor."""

    def generate_local_candidates(self, s0: Atoms, n: int = 20) -> list[Atoms]:
        """Generates candidates via random rattling."""
        candidates = []
        for i in range(n):
            c = s0.copy()  # type: ignore[no-untyped-call]
            c.rattle(stdev=0.05, seed=42 + i)
            candidates.append(c)
        return candidates
