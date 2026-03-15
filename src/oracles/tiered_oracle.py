from pathlib import Path

from ase import Atoms

from src.oracles.base import BaseOracle


class TieredOracle(BaseOracle):
    """Dynamic routing logic between a primary fast oracle and a slow fallback oracle based on uncertainty."""

    def __init__(
        self, primary_oracle: BaseOracle, fallback_oracle: BaseOracle, threshold: float
    ) -> None:
        self.primary_oracle = primary_oracle
        self.fallback_oracle = fallback_oracle
        self.threshold = threshold

    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """Routes structures to oracles based on uncertainty."""
        final_results = []
        fallback_queue = []

        # We need a way to map the original structures to their evaluated counterparts
        # to ensure the uncalculated original structure is passed to the fallback queue
        # as per SPEC: "append the original, uncalculated structure to a separate fallback_queue"

        # We keep the originals to pass them cleanly if they fail the threshold check
        original_structures = [atoms.copy() for atoms in structures]  # type: ignore[no-untyped-call]

        # 1. Primary evaluation
        evaluated_primary = self.primary_oracle.compute_batch(structures, calc_dir)

        # 2. Routing logic
        for i, evaluated_atoms in enumerate(evaluated_primary):
            mace_uncertainty = evaluated_atoms.info.get("mace_uncertainty")

            # If missing entirely, or strictly greater than / equal to threshold
            if mace_uncertainty is None or mace_uncertainty >= self.threshold:
                fallback_queue.append(original_structures[i])
            else:
                final_results.append(evaluated_atoms)

        # 3. Fallback execution
        if fallback_queue:
            evaluated_fallback = self.fallback_oracle.compute_batch(fallback_queue, calc_dir)
            final_results.extend(evaluated_fallback)

        return final_results
