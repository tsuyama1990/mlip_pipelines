import logging

from ase import Atoms

from src.core import AbstractGenerator
from src.domain_models.config import InterfaceTarget, StructureGeneratorConfig


class StructureGenerator(AbstractGenerator):
    """Generates localized candidate structures around an uncertain anchor."""

    def __init__(self, config: StructureGeneratorConfig) -> None:
        self.config = config

    def generate_local_candidates(self, s0: Atoms, n: int = 20) -> list[Atoms]:
        """Generates candidates via random rattling using streaming generation."""
        from collections.abc import Iterator

        if len(s0) > 10000:
            msg = "Structure is too large for rattling (OOM risk)."
            raise ValueError(msg)

        # Hard cap n to prevent memory exhaustion attacks
        n = min(n, 100)

        # Scale down n if the structure is massive to avoid OOM
        actual_n = n if len(s0) < 1000 else max(1, n // 10)

        def _generator() -> Iterator[Atoms]:
            for i in range(actual_n):
                c = s0.copy()  # type: ignore[no-untyped-call]
                c.rattle(stdev=self.config.stdev, seed=self.config.seed_base + i)
                yield c

        # In a fully streaming application this would yield directly,
        # but the trainer pipeline expects a list for selection sampling.
        # This wrapper explicitly bounds the generation safely.
        return list(_generator())

    def generate_interface(self, target: InterfaceTarget) -> Atoms:
        """Generates an interface structure based on an InterfaceTarget config."""
        from ase.build import bulk, stack
        from ase.data import chemical_symbols

        # Security: validate elements before passing to ASE
        valid_targets = self.config.valid_interface_targets

        for elem in [target.element1, target.element2]:
            if elem not in valid_targets and elem not in chemical_symbols:
                msg = f"Invalid or unsupported element target for interface generation: {elem}"
                raise ValueError(msg)

        logging.info(
            f"Generating interface between {target.element1} (face {target.face1}) "
            f"and {target.element2} (face {target.face2})."
        )

        try:
            # For simplicity, we create bulk representations and stack them
            # For FePt, we approximate it by building an Fe bulk, then a Pt bulk and stacking them?
            # Or just build an L1_0 ordered FePt? The tutorial explicitly mentions FePt vs MgO.
            # Let's generate a basic FePt structure and an MgO structure.

            # This is a simplified interface construction registry
            def _build_fept() -> Atoms:
                mat = bulk("Fe", crystalstructure="fcc", a=3.8) # type: ignore[no-untyped-call]
                mat[0].symbol = "Pt"
                return mat

            def _build_mgo() -> Atoms:
                return bulk("MgO", crystalstructure="rocksalt", a=4.21, basis=[[0, 0, 0], [0.5, 0.5, 0.5]]) # type: ignore[no-untyped-call]

            builders = {
                "FePt": _build_fept,
                "MgO": _build_mgo
            }

            def _build_generic(element: str) -> Atoms:
                if element in builders:
                    return builders[element]()
                return bulk(element) # type: ignore[no-untyped-call]

            mat1 = _build_generic(target.element1)
            mat2 = _build_generic(target.element2)

            # Adjust lattice parameters slightly to allow stacking without crashing
            # In a real scenario, this would use sophisticated mismatch analysis
            mat2.set_cell(mat1.get_cell(), scale_atoms=True) # type: ignore[no-untyped-call]
        except Exception as e:
            logging.exception("Failed to generate interface")
            msg = f"Interface generation failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return stack(mat1, mat2, axis=2, maxstrain=10.0)  # type: ignore[no-untyped-call]
