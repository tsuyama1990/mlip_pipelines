import os
import shutil
import subprocess
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from ase import Atoms
from ase.io import write
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from core.interfaces import AbstractPotential
from potentials.shifted_lj import ShiftedLennardJones

try:
    import pyace
except ImportError:
    pyace = None

class PyACEPotential(AbstractPotential):
    """
    PyACE potential wrapper with explicit Delta Learning support.
    Strategy: Delta Learning (Target = DFT - (LJ + E0)).
    """
    def __init__(self, model_path: str, atomic_energies: Dict[str, float], lj_params: Dict[str, float], device: str = "cpu"):
        self.model_path = model_path
        self.atomic_energies = atomic_energies or {}
        # lj_params: {sigma: ..., epsilon: ..., cutoff: ...}
        self.lj_params = lj_params or {}
        self.device = device
        self.ace_calc = None

        # Load if exists and pyace is available
        if pyace and os.path.exists(model_path):
             try:
                 self.ace_calc = pyace.PyACECalculator(model_path)
             except Exception as e:
                 logger.error(f"Failed to load ACE model from {model_path}: {e}")

    def _get_baseline_calc(self) -> ShiftedLennardJones:
        return ShiftedLennardJones(
            sigma=self.lj_params.get("sigma", 1.0),
            epsilon=self.lj_params.get("epsilon", 1.0),
            cutoff=self.lj_params.get("cutoff", 5.0)
        )

    def _compute_baseline(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """
        Compute V_base = V_LJ + sum(E0).
        """
        calc = self._get_baseline_calc()

        # Copy atoms to avoid modifying input and attach calculator
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc

        try:
            e_lj = atoms_copy.get_potential_energy()
            f_lj = atoms_copy.get_forces()
        except Exception as e:
            logger.warning(f"Baseline calculation failed: {e}")
            e_lj = 0.0
            f_lj = np.zeros((len(atoms), 3))

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

        # ACE Prediction
        e_ace = 0.0
        f_ace = np.zeros_like(f_base)
        s_ace = np.zeros((3, 3))

        if self.ace_calc:
            atoms_copy = atoms.copy()
            atoms_copy.calc = self.ace_calc
            try:
                e_ace = atoms_copy.get_potential_energy()
                f_ace = atoms_copy.get_forces()
                # Stress not supported/requested yet
            except Exception as e:
                logger.warning(f"ACE prediction failed: {e}")

        # Total
        return e_ace + e_base, f_ace + f_base, s_ace

    def train(self, training_data: List[Atoms], atomic_energies: Optional[Dict[str, float]] = None, energy_weight: float = 1.0, forces_weight: float = 10.0, **kwargs) -> None:
        """
        Train the ACE potential on residuals.
        """
        if atomic_energies:
            self.atomic_energies.update(atomic_energies)

        logger.info("Preparing training data for Delta Learning...")
        fitting_data = []
        elements = set()

        for atoms in training_data:
            elements.update(atoms.get_chemical_symbols())
            e_dft = atoms.get_potential_energy()
            f_dft = atoms.get_forces()

            # Calculate Baseline
            e_base, f_base = self._compute_baseline(atoms)

            # Calculate Target (Residual)
            target_e = e_dft - e_base
            target_f = f_dft - f_base

            a = atoms.copy()
            a.calc = None
            # Store as standard energy/forces for ASE write to pick up
            # Note: writing to extxyz usually uses Calculator results or info/arrays
            # We set info['energy'] and arrays['forces'] which are standard ASE conventions for 'cached' properties
            a.info['energy'] = target_e
            a.new_array('forces', target_f)
            fitting_data.append(a)

        # Write dataset
        train_file = "training_data.xyz"
        write(train_file, fitting_data, format="extxyz")

        # Generate input.yaml for Pacemaker
        # This is a minimal configuration
        input_conf = OmegaConf.create({
            "data": {
                "filename": train_file
            },
            "potential": {
                "basis_size": kwargs.get("basis_size", 500),
                "cutoff": self.lj_params.get("cutoff", 5.0), # Match LJ cutoff?
                "elements": sorted(list(elements))
            },
            "fit": {
                "loss": {
                    "kappa": forces_weight / energy_weight if energy_weight > 0 else 10.0
                },
                "weighting": {
                    "type": "energy_based" # Example
                }
            },
            "backend": {
                "evaluator": "tensorpot"
            }
        })

        OmegaConf.save(input_conf, "input.yaml")

        # Run Pacemaker
        cmd = ["pacemaker", "input.yaml"]
        logger.info(f"Running pacemaker: {' '.join(cmd)}")

        if shutil.which("pacemaker"):
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # Assume output is output_potential.yace
                output_model = "output_potential.yace"
                if os.path.exists(output_model):
                    shutil.move(output_model, self.model_path)
                    logger.success(f"Training successful. Model saved to {self.model_path}")

                    # Reload
                    if pyace:
                        self.ace_calc = pyace.PyACECalculator(self.model_path)
                else:
                    logger.error("Pacemaker finished but output_potential.yace not found.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Pacemaker failed: {e}")
                # Log stdout/stderr
                logger.error(f"Stdout: {e.stdout.decode() if e.stdout else ''}")
                logger.error(f"Stderr: {e.stderr.decode() if e.stderr else ''}")
                raise e
        else:
            logger.warning("Pacemaker executable not found. Skipping training execution (Mock Mode).")
            # For testing purposes, if in mock mode, we might want to create a dummy file?
            # But tests verify logic via mocks.

    def get_uncertainty(self, atoms: Atoms) -> np.ndarray:
        if not self.ace_calc:
            return np.ones(len(atoms))

        try:
            # Assuming compute_gamma
            gamma = self.ace_calc.compute_gamma(atoms)
            return np.array(gamma)
        except AttributeError:
            return np.zeros(len(atoms))

    def save(self, path: str) -> None:
        if self.model_path != path and os.path.exists(self.model_path):
             shutil.copy(self.model_path, path)
             self.model_path = path

    def load(self, path: str) -> None:
        self.model_path = path
        if pyace and os.path.exists(path):
            self.ace_calc = pyace.PyACECalculator(path)
