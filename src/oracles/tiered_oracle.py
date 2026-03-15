from pathlib import Path

from ase import Atoms

from src.oracles.base import BaseOracle


class TieredOracle(BaseOracle):
    """
    Dynamic routing logic between a primary fast oracle (like MACE)
    and a fallback high-fidelity oracle (like DFT) based on uncertainty.
    """

    def __init__(
        self,
        primary_oracle: BaseOracle,
        fallback_oracle: BaseOracle,
        threshold: float,
    ) -> None:
        self.primary_oracle = primary_oracle
        self.fallback_oracle = fallback_oracle
        self.threshold = threshold

    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """
        Routes structures to the primary oracle, checking uncertainty.
        If uncertainty meets or exceeds the threshold, or is missing,
        it evaluates them with the fallback oracle.
        """
        if not structures:
            return []

        # First evaluate everything with the primary oracle
        primary_results = self.primary_oracle.compute_batch(structures, calc_dir)

        final_results = []
        fallback_queue = []
        fallback_indices = []

        # Check uncertainty for routing
        for i, (orig_atom, result_atom) in enumerate(zip(structures, primary_results, strict=True)):
            uncertainty = result_atom.info.get("mace_uncertainty", float("inf"))
            if uncertainty < self.threshold:
                final_results.append(result_atom)
            else:
                fallback_queue.append(orig_atom.copy())  # copy original unannotated structure
                fallback_indices.append(i)

        # Evaluate uncertain structures with the fallback oracle
        if fallback_queue:
            fallback_results = self.fallback_oracle.compute_batch(
                fallback_queue,
                calc_dir / "fallback",
            )
            # Add to final results
            final_results.extend(fallback_results)

        return final_results
