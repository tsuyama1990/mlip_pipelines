import pytest
import sys
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock, patch
import importlib
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from carvers.chemistry import StoichiometryGuard
from potentials.factory import load_potential
from omegaconf import OmegaConf
from potentials.shifted_lj import ShiftedLennardJones

def test_stoichiometry_guard_basic():
    try:
        guard = StoichiometryGuard("NaCl")
    except ImportError:
        pytest.skip("pymatgen not installed")

    a1 = Atoms("NaCl", positions=[[0,0,0], [2,0,0]])
    assert guard.check(a1)

    a2 = Atoms("Na2Cl", positions=[[0,0,0], [1,0,0], [2,0,0]])
    assert not guard.check(a2)

def test_stoichiometry_guard_correction():
    try:
        guard = StoichiometryGuard("NaCl")
    except ImportError:
        pytest.skip("pymatgen not installed")

    # Na at 0 (center), Na at 10 (far), Cl at 2
    a3 = Atoms("NaNaCl", positions=[[0,0,0], [10,0,0], [2,0,0]])

    corrected, kept = guard.correct(a3, center_index=0)

    assert len(corrected) == 2
    assert corrected.get_chemical_formula() in ["NaCl", "ClNa"]
    assert 1 not in kept
    assert 0 in kept
    assert 2 in kept

def test_shifted_lj():
    # Sigma=1.0, Epsilon=1.0, Cutoff=2.0
    calc = ShiftedLennardJones(sigma=1.0, epsilon=1.0, cutoff=2.0)

    # At r=1.0 (min of unshifted LJ)
    # V_unshifted(1.0) = 4*1*(1 - 1) = 0? No!
    # V = 4*eps*((s/r)^12 - (s/r)^6).
    # r=1.0, s=1.0 -> 1^12 - 1^6 = 0.
    # Ah, standard LJ crosses 0 at sigma. Min is at 2^(1/6)*sigma.

    # At r=sigma=1.0, V_unshifted = 0.
    # V_shifted = 0 - V(rc).

    atoms = Atoms("Ar2", positions=[[0,0,0], [1.0,0,0]])
    atoms.calc = calc
    e = atoms.get_potential_energy()

    # Calculate expected V(rc)
    # rc=2.0. s/rc = 0.5.
    sr6 = 0.5**6 # 1/64
    sr12 = 0.5**12 # 1/4096
    v_rc = 4 * 1.0 * (sr12 - sr6)

    expected = 0.0 - v_rc
    assert np.isclose(e, expected)

    # At cutoff
    atoms_rc = Atoms("Ar2", positions=[[0,0,0], [2.0,0,0]])
    atoms_rc.calc = calc
    e_rc = atoms_rc.get_potential_energy()
    assert np.isclose(e_rc, 0.0)

def test_pyace_logic():
    mock_pyace = MagicMock()
    with patch.dict(sys.modules, {'pyace': mock_pyace}):
        import potentials.pyace_impl
        importlib.reload(potentials.pyace_impl)
        from potentials.pyace_impl import PyACEPotential

        pot = PyACEPotential("model.yace", {}, {"sigma":1.0, "epsilon":1.0, "cutoff":2.0})

        # Baseline check (Shifted LJ)
        atoms = Atoms("Ar2", positions=[[0,0,0], [2.0, 0, 0]])
        e_base, f_base = pot._compute_baseline(atoms)
        assert np.isclose(e_base, 0.0) # At cutoff

        # Predict
        pot.ace_calc = MagicMock()
        pot.ace_calc.get_potential_energy.return_value = 5.0
        pot.ace_calc.get_forces.return_value = np.zeros((2,3))

        e, f, s = pot.predict(atoms)
        assert np.isclose(e, 5.0 + 0.0)

def test_potential_factory_pyace():
    conf = OmegaConf.create({
        "potential": {
            "arch": "pyace",
            "pyace": {
                "model_path": "test.yace",
                "delta_learning": {
                    "enabled": True,
                    "lj_params": {"sigma": 1.0, "epsilon": 1.0, "cutoff": 5.0}
                }
            },
        },
        "device": "cpu"
    })

    pot = load_potential(conf)
    assert type(pot).__name__ == "PyACEPotential"
    assert pot.lj_params["sigma"] == 1.0
