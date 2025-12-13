import os
import sys

import numpy as np
import torch
from ase.build import bulk
from e3nn import o3
from loguru import logger
from mace.modules import ScaleShiftMACE
from mace.modules.blocks import RealAgnosticInteractionBlock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from src.carvers.box_carver import BoxCarver
from src.oracles.mock_oracle import MockOracle
from src.potentials.mace_impl import MacePotential

# Setup Logger
logger.remove()
logger.add(sys.stdout, format="{time} - {level} - {message}", level="INFO")

def create_dummy_model(path):
    if os.path.exists(path):
        return

    model_config = dict(
        r_max=4.0,
        num_bessel=3,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=RealAgnosticInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps('8x0e'),
        MLP_irreps=o3.Irreps('8x0e'),
        atomic_energies=np.array([0.0, 0.0]),
        avg_num_neighbors=1.0,
        atomic_numbers=[1, 29], # H, Cu
        correlation=1,
        gate=torch.nn.functional.silu,
    )

    model = ScaleShiftMACE(
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        **model_config
    )
    torch.save(model, path)
    logger.info(f"Saved dummy model to {path}")

def run_mock_active_learning():
    model_path = "temp_foundation.model"
    create_dummy_model(model_path)

    # 1. Integration
    logger.info("Initializing Components...")
    pot = MacePotential(model_path)
    oracle = MockOracle()

    # 2. Loop
    n_iterations = 5
    logger.info(f"Starting Active Learning Loop for {n_iterations} iterations.")

    for i in range(n_iterations):
        logger.info(f"--- Iteration {i+1} ---")

        # Generate random structure
        atoms = bulk('Cu', cubic=True) * (2, 2, 2)
        atoms.rattle(stdev=0.1 + (i * 0.05), seed=i) # Increasing disorder

        # Predict
        uncertainties = pot.get_uncertainty(atoms)
        max_u_idx = np.argmax(uncertainties)
        max_u_val = uncertainties[max_u_idx]

        logger.info(f"Max Uncertainty: {max_u_val:.4f} at atom {max_u_idx}")

        # Carve High Uncertainty
        carver = BoxCarver(atoms, center_index=max_u_idx, box_vector=6.0)
        cluster = carver.carve(skin_depth=1)
        logger.info(f"Carved cluster with {len(cluster)} atoms.")

        # Label (Mock)
        labeled_cluster = oracle.compute(cluster)

        # Compute Error (Mock validation)
        # We compare the potential's prediction on the labeled cluster vs the oracle's label
        pred_e, pred_f, _ = pot.predict(labeled_cluster)
        true_e = labeled_cluster.get_potential_energy()
        error = abs(pred_e - true_e)
        logger.info(f"Prediction Error (Energy): {error:.4f} eV")

        # Retrain
        # We must ensure the cluster has the calculated results attached for training
        # MacePotential.train uses get_potential_energy() from atoms
        pot.train([labeled_cluster], energy_weight=1.0, forces_weight=1.0)

        logger.info(f"Retraining complete. New u_max: {pot.u_max:.4f}")

if __name__ == "__main__":
    run_mock_active_learning()
