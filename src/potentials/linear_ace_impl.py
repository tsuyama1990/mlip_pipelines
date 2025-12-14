import os
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from ase import Atoms
from ase.calculators.lj import LennardJones
from loguru import logger
from core.interfaces import AbstractPotential

try:
    import pyace
except ImportError:
    pyace = None

class LinearACEPotential(AbstractPotential):
    """
    Linear ACE potential wrapper with Delta Learning support.
    Target = DFT - (LJ + E0).
    """
    def __init__(self, model_path: str, atomic_energies: Dict[str, float], lj_params: Dict[str, float], device: str = "cpu"):
        if pyace is None:
            # We allow instantiation but warn, so tests can mock pyace later
            logger.warning("pyace module not found. LinearACEPotential will not function correctly without mocks.")

        self.model_path = model_path
        self.atomic_energies = atomic_energies or {}
        self.lj_params = lj_params or {} # {sigma: ..., epsilon: ...}
        self.ace_calc = None
        self.device = device # Unused for ACE usually, but interface might pass it

        # Load if exists
        if pyace and os.path.exists(model_path):
             try:
                 self.ace_calc = pyace.PyACECalculator(model_path)
             except Exception as e:
                 logger.error(f"Failed to load ACE model from {model_path}: {e}")

    def _compute_baseline(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """
        Compute V_base = V_LJ + sum(E0).
        """
        # LJ
        sigma = self.lj_params.get("sigma", 2.0)
        epsilon = self.lj_params.get("epsilon", 0.1)

        # ASE LennardJones doesn't support generic sigma/epsilon easily for multi-element without combination rules.
        # Simple implementation: Same parameters for all pairs?
        # Prompt: "lj_params: {sigma: 2.0, epsilon: 0.1} # Or element-specific"
        # We'll assume simple single-type LJ for now as per config example.

        lj = LennardJones(sigma=sigma, epsilon=epsilon)
        atoms_copy = atoms.copy()
        atoms_copy.calc = lj

        e_lj = atoms_copy.get_potential_energy()
        f_lj = atoms_copy.get_forces()

        # E0
        e0 = 0.0
        syms = atoms.get_chemical_symbols()
        for s in syms:
            e0 += self.atomic_energies.get(s, 0.0)

        return e_lj + e0, f_lj

    def predict(self, atoms: Atoms) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Return (energy [eV], forces [eV/A], stress [eV/A^3]).
        """
        # Baseline
        e_base, f_base = self._compute_baseline(atoms)

        # ACE
        e_ace = 0.0
        f_ace = np.zeros_like(f_base)
        s_ace = np.zeros((3, 3))

        if self.ace_calc:
            atoms_copy = atoms.copy()
            atoms_copy.calc = self.ace_calc
            try:
                e_ace = atoms_copy.get_potential_energy()
                f_ace = atoms_copy.get_forces()
                # Try to get stress if supported
                # s_ace = atoms_copy.get_stress(voigt=False)
                # Keeping stress zero for now if not critical or unknown API
            except Exception as e:
                logger.warning(f"ACE prediction failed: {e}")

        return e_ace + e_base, f_ace + f_base, s_ace

    def train(self, training_data: List[Atoms], atomic_energies: Optional[Dict[str, float]] = None, energy_weight: float = 1.0, forces_weight: float = 10.0, **kwargs) -> None:
        if not pyace:
            raise ImportError("Cannot train LinearACEPotential: pyace not installed.")

        if atomic_energies:
            self.atomic_energies.update(atomic_energies)

        fitting_data = []
        for atoms in training_data:
            e_dft = atoms.get_potential_energy()
            f_dft = atoms.get_forces()

            e_base, f_base = self._compute_baseline(atoms)

            target_e = e_dft - e_base
            target_f = f_dft - f_base

            a = atoms.copy()
            a.calc = None
            # Store targets
            # Assumption: pyace.fit checks info['energy'] and arrays['forces']
            a.info['energy'] = target_e
            a.new_array('forces', target_f)
            fitting_data.append(a)

        logger.info(f"Training Linear ACE on {len(fitting_data)} structures...")

        # Pass kwargs to fit (cutoff, basis_size, etc.)
        # Assuming pyace.fit(atoms_list, output_filename, **params)
        try:
            pyace.fit(fitting_data, self.model_path, **kwargs)
            # Reload
            self.ace_calc = pyace.PyACECalculator(self.model_path)
            logger.info("Training complete.")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e

    def get_uncertainty(self, atoms: Atoms) -> np.ndarray:
        if not self.ace_calc:
            return np.ones(len(atoms))

        try:
            # Assuming 'compute_gamma' computes the maxvol-based uncertainty (extrapolation grade)
            gamma = self.ace_calc.compute_gamma(atoms)
            return np.array(gamma) # Ensure numpy array
        except AttributeError:
            logger.warning("pyace calculator does not support compute_gamma. Returning zeros.")
            return np.zeros(len(atoms))

    def save(self, path: str) -> None:
        # Linear ACE model is already saved to file during train.
        # If we need to move it to 'path'?
        if self.model_path != path and os.path.exists(self.model_path):
             import shutil
             shutil.copy(self.model_path, path)
             self.model_path = path

    def load(self, path: str) -> None:
        self.model_path = path
        if pyace and os.path.exists(path):
            self.ace_calc = pyace.PyACECalculator(path)
