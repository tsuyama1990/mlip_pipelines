from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.units import fs
from loguru import logger
from omegaconf import DictConfig

from carvers.box_carver import BoxCarver
from core.exceptions import OracleComputationError
from core.interfaces import AbstractOracle, AbstractPotential
from engines.lammps_mace import LammpsMaceDriver

# Assuming Generator interface exists or we pass it dynamically
# Cycle 2 memory says "src/generators/adapter.py" but prompt calls it StructureGenerator.
# I will assume the orchestrator receives an object satisfying a generator interface.

class ActiveLearningOrchestrator:
    def __init__(
        self,
        config: DictConfig,
        potential: AbstractPotential,
        oracle: AbstractOracle,
        generator: Any  # Duck-typed generator
    ):
        self.config = config
        self.potential = potential
        self.oracle = oracle
        self.generator = generator

        self.dataset: list[Atoms] = []
        self.cycle_count = 0
        self.experiment_dir = Path(config.get("experiment", {}).get("work_dir", "."))
        self.experiment_dir.mkdir(exist_ok=True, parents=True)

    def run_loop(self):
        """Execute the full AL loop."""
        max_cycles = self.config.experiment.max_cycles

        # Phase 0: Initialization
        logger.info("PHASE 0: Bootstrapping")
        # Generate initial pool
        # Assuming generator has generate_initial_pool(n)
        seeds = self.generator.generate_initial_pool(n=5)
        logger.info(f"Generated {len(seeds)} seed structures.")

        labeled_seeds = self._label_candidates(seeds)
        if labeled_seeds:
            self._update_potential(labeled_seeds)
        else:
            logger.error("Failed to label any seed structures. Aborting.")
            return

        while self.cycle_count < max_cycles:
            self.cycle_count += 1
            logger.info(f"=== CYCLE {self.cycle_count} START ===")

            # Phase 1: Exploration
            candidates = self.phase_exploration()

            # Phase 2: Active Learning
            if not candidates:
                logger.info("No uncertain structures found. Continuing exploration...")
                # In a real loop, we might want to restart MD with higher temp or different seeds
                continue

            logger.info(f"PHASE 2: Active Learning with {len(candidates)} candidates")

            # Label
            labeled_data = self._label_candidates(candidates)

            # Train
            if labeled_data:
                self._update_potential(labeled_data)
            else:
                logger.warning("No candidates were successfully labeled.")

        logger.info("Active Learning Loop Completed.")

    def phase_exploration(self) -> list[Atoms]:
        """
        Run MD with MACE. Monitor Uncertainty.
        Returns a list of 'carved' clusters (ready for labeling).
        """
        candidates = []

        # Use the latest structures from dataset as starting points for MD?
        # Or generate new ones? Prompt says: "Run MD... Monitor Uncertainty"
        # Let's pick a few random structures from the dataset to explore from.
        if not self.dataset:
            return []

        # Pick 1 random structure to explore
        start_idx = np.random.randint(0, len(self.dataset))
        atoms = self.dataset[start_idx].copy()

        # MD Parameters
        steps = self.config.exploration.md_steps
        temp = self.config.exploration.temperature

        # Get threshold ratio from config, default to 2.0 if missing
        threshold = self.config.get("active_learning", {}).get("threshold_ratio",
                    self.config.get("exploration", {}).get("threshold", 2.0))

        # Setup LAMMPS Driver
        driver = LammpsMaceDriver(self.potential, threshold=threshold)

        # Define standard NVT script
        script = f"fix 1 all nvt temp {temp} {temp} 0.1"

        logger.info(f"Starting MD exploration on structure {start_idx} for {steps} steps...")

        try:
            final_atoms, status = driver.run_md(atoms, script, steps)

            if status == "UNCERTAIN":
                logger.info("Uncertainty interrupt triggered.")

                # Check uncertainty to find center (redundant check, but safe)
                uncertainties = self.potential.get_uncertainty(final_atoms)
                center_idx = int(np.argmax(uncertainties))
                box_size = self.config.experiment.box_size

                try:
                    # Instantiate BoxCarver for the specific cut
                    carver = BoxCarver(final_atoms, center_idx, box_size)
                    cluster = carver.carve()
                    candidates.append(cluster)
                except Exception as e:
                    logger.warning(f"Carving failed: {e}")

            elif status == "FINISHED":
                logger.info("Exploration converged without uncertainty. Restarting exploration...")
                # We return empty list so the loop continues
                return []

        except Exception as e:
            logger.error(f"MD failed: {e}")
            return []

        return candidates

    def _label_candidates(self, candidates: list[Atoms]) -> list[Atoms]:
        """
        Send to Oracle. Handle failures robustly.
        """
        labeled = []
        for i, atoms in enumerate(candidates):
            logger.info(f"Labeling candidate {i+1}/{len(candidates)}...")
            try:
                result = self.oracle.compute(atoms)
                labeled.append(result)
            except OracleComputationError as e:
                logger.warning(f"Candidate discarded due to DFT failure: {e}")
                # Do NOT crash. Just skip.
        return labeled

    def _update_potential(self, new_data: list[Atoms]):
        """
        Add to dataset, Save, Retrain.
        """
        logger.info(f"Adding {len(new_data)} new structures to dataset.")
        self.dataset.extend(new_data)

        # Save dataset backup
        # (Simplified for now, maybe pickling or extended XYZ)
        # ase.io.write(self.experiment_dir / f"dataset_cycle_{self.cycle_count}.xyz", self.dataset)

        # Train
        logger.info("Retraining potential...")
        self.potential.train(
            self.dataset,
            energy_weight=self.config.training.energy_weight,
            forces_weight=self.config.training.forces_weight
        )

        # Save model
        model_path = self.experiment_dir / f"model_cycle_{self.cycle_count}.pt"
        self.potential.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
