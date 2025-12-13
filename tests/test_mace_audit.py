import numpy as np
import pytest
from ase import Atoms
from src.potentials.mace_impl import MacePotential
import torch
import os
from ase.calculators.singlepoint import SinglePointCalculator

def test_head_only_frozen_check(minimal_mace_model_path):
    """
    Auditor Check: Verify that the backbone is truly frozen during training.
    """
    pot = MacePotential(minimal_mace_model_path)

    # Capture backbone weights before training
    # MockMACE has `backbone` linear layer.
    backbone_weight_before = pot.model.backbone.weight.clone()
    readout_weight_before = pot.model.readouts[0].weight.clone()

    # Train
    train_atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=True)
    calc = SinglePointCalculator(train_atoms, energy=-10.0, forces=np.zeros((2, 3)))
    train_atoms.calc = calc
    pot.train([train_atoms])

    # Capture after
    backbone_weight_after = pot.model.backbone.weight
    readout_weight_after = pot.model.readouts[0].weight

    # Assert backbone is frozen (identical)
    assert torch.equal(backbone_weight_before, backbone_weight_after), "Backbone weights changed!"

    # Assert readout changed (trained)
    assert not torch.equal(readout_weight_before, readout_weight_after), "Readout weights did not change!"

def test_persistence_check(minimal_mace_model_path, tmp_path):
    """
    Auditor Check: Save/Load and verify `inv_covariance` and uncertainty are restored.
    """
    pot = MacePotential(minimal_mace_model_path)

    # Train to generate covariance
    train_atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=True)
    calc = SinglePointCalculator(train_atoms, energy=-10.0, forces=np.zeros((2, 3)))
    train_atoms.calc = calc
    pot.train([train_atoms])

    # Compute uncertainty before save
    u_before = pot.get_uncertainty(train_atoms)
    cov_before = pot.inv_covariance.clone()

    # Save
    save_path = tmp_path / "audit_model.pt"
    pot.save(save_path)

    # Load into new instance
    pot2 = MacePotential(str(save_path))

    # Check covariance restoration
    assert pot2.inv_covariance is not None
    assert torch.allclose(cov_before, pot2.inv_covariance)

    # Check uncertainty consistency
    u_after = pot2.get_uncertainty(train_atoms)
    assert np.allclose(u_before, u_after)

def test_singular_matrix_handling(minimal_mace_model_path):
    """
    Auditor Check: Test stability with very few data points (N < D).
    """
    pot = MacePotential(minimal_mace_model_path)

    # Dimension is 16 in MockMACE.
    # Provide 1 atom. N=1.
    train_atoms = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    calc = SinglePointCalculator(train_atoms, energy=-5.0, forces=np.zeros((1, 3)))
    train_atoms.calc = calc

    # Should not crash thanks to regularization
    pot.train([train_atoms])

    assert pot.inv_covariance is not None
    assert not torch.isnan(pot.inv_covariance).any()
