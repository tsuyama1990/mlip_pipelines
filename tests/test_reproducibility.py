import numpy as np
import pytest
from config.settings import Settings, GeneratorSettings
from generators.adapter import ExternalGeneratorAdapter

class TestReproducibility:
    def test_bit_identical_with_same_seed(self):
        """MASD 2.3: Same seed → identical output"""
        # We need to mock nnp_gen because we can't easily rely on external code logic
        # that might involve complex random states, but we want to verify OUR adapter passes the seed correctly.
        # However, checking bit-identical output implies the external generator respects the seed.
        # Assuming external generator works as intended, we check end-to-end.

        # If external generator is deterministic given a seed, this passes.

        settings1 = Settings(generator=GeneratorSettings(target_element="Si"))
        settings1.random_seed = 12345

        settings2 = Settings(generator=GeneratorSettings(target_element="Si"))
        settings2.random_seed = 12345

        adapter1 = ExternalGeneratorAdapter(settings1.generator, seed=settings1.random_seed)
        adapter2 = ExternalGeneratorAdapter(settings2.generator, seed=settings2.random_seed)

        atoms1 = adapter1.generate()
        atoms2 = adapter2.generate()

        assert np.array_equal(atoms1.positions, atoms2.positions), \
            "Same seed must produce bit-identical structures"
