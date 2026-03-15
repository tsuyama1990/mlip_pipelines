from pathlib import Path
from typing import Any

from ase import Atoms

from src.domain_models.config import ProjectConfig
from src.oracles.base import BaseOracle


class MACEManager(BaseOracle):
    """
    High-performance wrapper for MACE foundation model.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._calc: Any = None

    def _get_mace_calculator(self) -> Any:
        """Lazy load the MACE calculator."""
        if self._calc is None:
            # We delay importing to avoid high import time/memory unless used
            from mace.calculators import mace_mp  # type: ignore[import-untyped]

            # In MACE, mace_mp returns a MACECalculator if requested model is mace-mp-X
            self._calc = mace_mp(
                model=self.config.distillation_config.mace_model_path,
                dispersion=False,
                default_dtype="float32",
                device="cpu",  # Assume CPU unless configured otherwise (could be expanded)
                return_raw_model=False,
            )
        return self._calc

    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """Runs MACE on a batch of structures, extracting energy, forces and uncertainty."""
        calc = self._get_mace_calculator()
        results = []

        for atoms in structures:
            # Important: copy atoms to prevent side effects
            annotated_atoms = atoms.copy()  # type: ignore[no-untyped-call]
            annotated_atoms.calc = calc

            # trigger calculation
            energy = annotated_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            forces = annotated_atoms.get_forces()  # type: ignore[no-untyped-call]

            # The uncertainty metric is returned directly in calc.results dictionary
            # MACE typically uses 'node_energy_variance' or 'mace_uncertainty'
            # In MACE, compute uncertainty is stored as node_energy_variance and energy_variance
            # Sometimes 'mace_uncertainty' might be output explicitly, or 'energy_var'

            uncertainty = None
            if "mace_uncertainty" in calc.results:
                uncertainty = float(calc.results["mace_uncertainty"])
            elif "node_energy_variance" in calc.results:
                # If it's per-node, we take the max node variance as the structure uncertainty
                import numpy as np

                uncertainty = float(np.max(calc.results["node_energy_variance"]))
            elif "energy_variance" in calc.results:
                uncertainty = float(calc.results["energy_variance"])
            elif "energy_var" in calc.results:
                uncertainty = float(calc.results["energy_var"])
            elif "error" in calc.results:
                uncertainty = float(calc.results["error"])  # fallback for some mock configs

            if uncertainty is None:
                msg = f"MACE calculator failed to produce an uncertainty metric. Available keys: {list(calc.results.keys())}"
                raise ValueError(msg)

            # Assign to dictionary as strictly required by design architecture
            annotated_atoms.info["energy"] = energy
            annotated_atoms.arrays["forces"] = forces.copy()
            annotated_atoms.info["mace_uncertainty"] = uncertainty

            results.append(annotated_atoms)

        return results
