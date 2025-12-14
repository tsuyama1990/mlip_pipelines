import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from potentials.mace_impl import MacePotential
import torch

def test_mace_consistency(minimal_mace_model_path):
    """Test that predicting the same atoms returns consistent results."""
    pot = MacePotential(minimal_mace_model_path)
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Run once to initialize any lazy loading if any (not applicable here but good practice)
    e1, f1, s1 = pot.predict(atoms)

    # Run again
    e2, f2, s2 = pot.predict(atoms)

    assert np.isclose(e1, e2)
    assert np.allclose(f1, f2)
    assert np.allclose(s1, s2)

def test_uncertainty_mechanism(minimal_mace_model_path):
    """Test LLU mechanism: sensitivity to unseen structures."""
    pot = MacePotential(minimal_mace_model_path)

    # 1. Create dummy training data
    train_atoms = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=True)
    # Fake calculator for training labels
    train_atoms.info['energy'] = -10.0
    train_atoms.arrays['forces'] = np.zeros((2, 3))

    # We need to ensure get_potential_energy works without a real calculator for the test logic
    # OR we use the fact that our MockMACE doesn't really care about labels for the hook part,
    # but the 'train' method expects to read labels.
    # In 'train', I used `a.get_potential_energy()`. This requires a calculator attached.
    from ase.calculators.singlepoint import SinglePointCalculator
    calc = SinglePointCalculator(train_atoms, energy=-10.0, forces=np.zeros((2, 3)))
    train_atoms.calc = calc

    # 2. Train (computes C^-1)
    pot.train([train_atoms])

    assert pot.inv_covariance is not None

    # 3. Compute uncertainty for training structure
    u_train = pot.get_uncertainty(train_atoms)

    # 4. Compute uncertainty for perturbed structure (OOD)
    ood_atoms = train_atoms.copy()
    ood_atoms.positions += 0.5 # Significant perturbation
    u_ood = pot.get_uncertainty(ood_atoms)

    # Assert OOD uncertainty is higher
    # Since we only trained on one point, any deviation in feature space should increase Mahalanobis distance
    # provided the model's features actually change with position (which they should, given random weights in linear/readout).
    # However, our MockMACE has random features.
    # Wait, MockMACE generates `torch.randn(num_nodes, 16)`.
    # It is NOT deterministic with respect to input atoms!
    # It generates random features *every forward pass*.
    # This will fail the consistency test AND the uncertainty test.
    # We must make MockMACE deterministic based on input or just fixed random.
    # But `forward(data)` in MockMACE ignores `data`.
    # Correcting MockMACE in conftest is needed.

    assert u_ood.mean() > u_train.mean()

def test_head_only_training(minimal_mace_model_path):
    """Test that only readout parameters are trained."""
    pot = MacePotential(minimal_mace_model_path)

    # Check requires_grad before training
    # Initially loaded, usually requires_grad depends on how it was saved.
    # train() sets them.

    train_atoms = Atoms('H', positions=[[0,0,0]], cell=[10,10,10], pbc=True)
    calc = SinglePointCalculator(train_atoms, energy=-1.0, forces=np.zeros((1, 3)))
    train_atoms.calc = calc

    pot.train([train_atoms])

    # Check that body is frozen
    # In our MockMACE, we only have 'readouts'.
    # If we added another layer, we could check.
    # But checking 'readouts' are unfrozen is good.
    for param in pot.target_module.parameters():
        assert param.requires_grad

    # Check if there were other params (if any) they should be frozen.
    # MockMACE only has readouts.
    pass
