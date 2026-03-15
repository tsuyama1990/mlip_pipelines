from pathlib import Path

from ase import Atoms

from src.domain_models.config import DistillationConfig
from src.oracles.base import BaseOracle


class MACEManager(BaseOracle):
    """Orchestrates the MACE foundation model for rapid evaluation."""

    def __init__(self, config: DistillationConfig) -> None:
        self.config = config

        # We instantiate the MACE calculator using the configured model path.
        # This wrapper assumes mace is installed and available.
        from mace.calculators.mace_mp import mace_mp  # type: ignore[import-untyped]

        self.calculator = mace_mp(
            model=self.config.mace_model_path,
            dispersion=False,
            default_dtype="float64",
            device="cpu"  # Assuming CPU for general compatibility, can be optimized later
        )

    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """Runs MACE computation on a batch of structures."""
        results = []
        for atoms in structures:
            atoms_copy = atoms.copy()  # type: ignore[no-untyped-call]
            atoms_copy.calc = self.calculator

            try:
                # Calculate properties. MACE calculator usually handles forces and energy
                energy = atoms_copy.get_potential_energy()  # type: ignore[no-untyped-call]
                forces = atoms_copy.get_forces()  # type: ignore[no-untyped-call]

                # Try to extract uncertainty. Depending on the MACE version and model (e.g., ensemble),
                # it might be stored in different keys. mace_mp often returns it in results.
                # If using an ensemble, it might be 'energy_var' or 'forces_var' or 'mace_uncertainty'
                # We will check common keys in calculator results.
                calc_results = atoms_copy.calc.results

                uncertainty = None

                if "mace_uncertainty" in calc_results:
                    uncertainty = calc_results["mace_uncertainty"]
                elif "energy_var" in calc_results:
                    uncertainty = calc_results["energy_var"]
                elif "forces_var" in calc_results:
                    # If forces_var is an array, take the max or mean as a scalar uncertainty metric
                    import numpy as np
                    uncertainty = float(np.max(calc_results["forces_var"]))

                if uncertainty is None:
                    self._raise_missing_uncertainty()

                # Write back to info and arrays to standardize output as requested by SPEC
                atoms_copy.info["energy"] = energy
                atoms_copy.arrays["forces"] = forces
                atoms_copy.info["mace_uncertainty"] = float(uncertainty)

                results.append(atoms_copy)

            except Exception as e:
                # If any step fails, we must wrap it or re-raise according to constraints
                if isinstance(e, ValueError) and "failed to extract uncertainty" in str(e):
                    raise
                msg = f"MACEManager failed to compute properties for structure: {e}"
                raise ValueError(msg) from e

        return results

    def _raise_missing_uncertainty(self) -> None:
        msg = "MACEManager failed to extract uncertainty metric from MACE calculator results."
        raise ValueError(msg)
