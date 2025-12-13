from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from ase import Atoms
from ase.build import bulk

class TestPropertyBased:
    @given(
        translation=arrays(
            dtype=np.float64,
            shape=(3,),
            elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)
        )
    )
    def test_translation_invariance(self, translation):
        """MASD 5.1: Translation preserves relative distances"""
        atoms1 = bulk('Si', cubic=True)
        atoms2 = atoms1.copy()
        atoms2.translate(translation)

        dist1 = atoms1.get_all_distances(mic=False)
        dist2 = atoms2.get_all_distances(mic=False)

        assert np.allclose(dist1, dist2, atol=1e-10)
