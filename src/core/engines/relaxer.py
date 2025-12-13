import time
from typing import Dict, Any
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS, BFGS, FIRE
from loguru import logger
from src.config.settings import Settings

# Map optimizer string to class
OPTIMIZERS = {
    "LBFGS": LBFGS,
    "BFGS": BFGS,
    "FIRE": FIRE
}

class StructureRelaxer:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Calculator is set externally or passed?
        # Usually easier to pass calculator to run() or set it here if Factory is used.
        # Design: main_cli instantiates calc and assigns to atoms. Relaxer just relaxes atoms.
        pass

    def run(self, atoms: Atoms, run_id: str) -> Dict[str, Any]:
        """
        Run structure optimization.
        """
        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached.")

        logger.info(f"Starting relaxation for Run ID: {run_id}")

        # 1. Pre-Optimization Info
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        max_force_init = np.sqrt((initial_forces**2).sum(axis=1).max())

        logger.info(f"Initial Energy: {initial_energy:.4f} eV")
        logger.info(f"Initial Max Force: {max_force_init:.4f} eV/A")

        # 2. Setup Optimizer
        opt_class = OPTIMIZERS.get(self.settings.relax.optimizer, LBFGS)

        # Trajectory file (optional, keeps history)
        # We can use a list to store it in memory for the result dict
        trajectory = []
        def traj_writer(atoms=atoms):
            trajectory.append(atoms.copy())

        opt = opt_class(atoms, logfile=None) # We handle logging
        opt.attach(traj_writer, interval=1)

        # 3. Run Optimization
        start_time = time.time()
        converged = False
        final_steps = 0

        try:
            converged = opt.run(fmax=self.settings.relax.fmax, steps=self.settings.relax.steps)
            final_steps = opt.get_number_of_steps()
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Don't raise, return what we have

        end_time = time.time()
        duration = end_time - start_time

        # 4. Post-Optimization Info
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        max_force_final = np.sqrt((final_forces**2).sum(axis=1).max())

        logger.info(f"Final Energy: {final_energy:.4f} eV")
        logger.info(f"Final Max Force: {max_force_final:.4f} eV/A")
        logger.info(f"Steps: {final_steps}, Converged: {converged}, Time: {duration:.2f}s")

        result = {
            "run_id": run_id,
            "converged": converged,
            "steps": final_steps,
            "duration_seconds": duration,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "max_force_initial": max_force_init,
            "max_force_final": max_force_final,
            "final_structure": atoms.copy(),
            "trajectory": trajectory
        }

        return result
