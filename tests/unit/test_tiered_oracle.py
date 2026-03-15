from pathlib import Path

from ase import Atoms

from src.oracles.base import BaseOracle
from src.oracles.tiered_oracle import TieredOracle


class MockPrimaryOracle(BaseOracle):
    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        results = []
        for i, atom in enumerate(structures):
            annotated = atom.copy()
            # Interleave uncertain and certain predictions
            if i % 2 == 0:
                annotated.info["mace_uncertainty"] = 0.01  # below 0.05
            else:
                annotated.info["mace_uncertainty"] = 0.10  # above 0.05

            # Special case for missing
            if i == 4:
                annotated.info.pop("mace_uncertainty")
            results.append(annotated)
        return results


class MockFallbackOracle(BaseOracle):
    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        results = []
        for atom in structures:
            annotated = atom.copy()
            annotated.info["dft_evaluated"] = True
            results.append(annotated)
        return results


def test_tiered_oracle_routing(tmp_path):
    primary_oracle = MockPrimaryOracle()
    fallback_oracle = MockFallbackOracle()

    threshold = 0.05
    tiered_oracle = TieredOracle(
        primary_oracle=primary_oracle,
        fallback_oracle=fallback_oracle,
        threshold=threshold,
    )

    structures = [Atoms("Fe") for _ in range(10)]

    # 0, 2, 6, 8 will be below threshold (0.01) -> final
    # 1, 3, 5, 7, 9 will be above threshold (0.10) -> fallback
    # 4 will be missing uncertainty -> fallback

    results = tiered_oracle.compute_batch(structures, tmp_path)

    assert len(results) == 10

    # Fallback oracle evaluation flags
    fallback_evaluated = [1 for r in results if r.info.get("dft_evaluated")]

    # Exactly 6 should hit fallback
    assert len(fallback_evaluated) == 6

    # Exactly 4 should hit primary fast-path only
    fast_path = [1 for r in results if not r.info.get("dft_evaluated")]
    assert len(fast_path) == 4
