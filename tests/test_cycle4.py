import pytest
import sys
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch
import importlib

# Ensure src is in path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from carvers.chemistry import StoichiometryGuard
from potentials.factory import load_potential
from omegaconf import OmegaConf

def test_stoichiometry_guard_basic():
    try:
        guard = StoichiometryGuard("NaCl")
    except ImportError:
        pytest.skip("pymatgen not installed")

    # 1. Perfect match
    a1 = Atoms("NaCl", positions=[[0,0,0], [2,0,0]])
    assert guard.check(a1)

    # 2. Excess Na
    a2 = Atoms("Na2Cl", positions=[[0,0,0], [1,0,0], [2,0,0]])
    assert not guard.check(a2)

def test_stoichiometry_guard_correction():
    try:
        guard = StoichiometryGuard("NaCl")
    except ImportError:
        pytest.skip("pymatgen not installed")

    # Na at 0 (center), Na at 10 (far), Cl at 2
    # Excess Na -> Remove furthest Na (at 10)
    a3 = Atoms("NaNaCl", positions=[[0,0,0], [10,0,0], [2,0,0]])

    # Center index 0 (Na at 0,0,0)
    corrected, kept = guard.correct(a3, center_index=0)

    assert len(corrected) == 2
    # Formula order depends on ASE version/settings (NaCl vs ClNa)
    assert corrected.get_chemical_formula() in ["NaCl", "ClNa"]
    # Indices in original: 0=Na, 1=Na(far), 2=Cl.
    # We expect 1 to be removed.
    assert 1 not in kept
    assert 0 in kept
    assert 2 in kept

def test_linear_ace_logic():
    # Mock pyace
    mock_pyace = MagicMock()

    with patch.dict(sys.modules, {'pyace': mock_pyace}):
        # Reload to pick up mock
        import potentials.linear_ace_impl
        importlib.reload(potentials.linear_ace_impl)
        from potentials.linear_ace_impl import LinearACEPotential

        # Init
        pot = LinearACEPotential("model.yace", {}, {"sigma":2.0, "epsilon":0.1})

        # Test Baseline (LJ)
        # Pair of atoms at distance 2.0 (sigma)
        # LJ at sigma is 0? No, 4*eps*((1)^12 - (1)^6) = 0.
        atoms = Atoms("Ar2", positions=[[0,0,0], [2.0, 0, 0]])

        e, f, s = pot.predict(atoms)
        # LJ at r=sigma is 0. But due to cutoff corrections, it might deviate slightly.
        assert np.abs(e) < 0.01

        # At distance < sigma, should be positive (repulsive)
        atoms_close = Atoms("Ar2", positions=[[0,0,0], [1.8, 0, 0]])
        e_close, _, _ = pot.predict(atoms_close)
        assert e_close > 0.0

        # At r_min = 2^(1/6)*sigma approx 1.122*2.0 = 2.244
        atoms_min = Atoms("Ar2", positions=[[0,0,0], [2.244, 0, 0]])
        e_min, _, _ = pot.predict(atoms_min)
        # Should be approx -epsilon (-0.1)
        assert np.isclose(e_min, -0.1, atol=0.01)

def test_potential_factory():
    conf = OmegaConf.create({
        "potential": {
            "arch": "linear_ace",
            "linear_ace": {"model_path": "test.yace"},
            "delta_learning": {"lj_params": {"sigma": 1.0, "epsilon": 1.0}}
        },
        "device": "cpu"
    })

    # Reload factory to ensure it sees the reloaded linear_ace_impl from previous test if shared?
    # Pytest runs in one process usually.
    # If previous test patched sys.modules, context manager exit should restore it.
    # But importlib.reload might have permanently affected potentials.linear_ace_impl module object in memory.

    # We'll just run factory. If LinearACEPotential imports pyace and fails, it logs warning but instantiates.
    # Since we are not patching here, pyace will be None (real env).
    # LinearACEPotential.__init__ checks if pyace is None.

    pot = load_potential(conf)
    # Check class name to avoid reload issues in test suite
    assert type(pot).__name__ == "LinearACEPotential"
