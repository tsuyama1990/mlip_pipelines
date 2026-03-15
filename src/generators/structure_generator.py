import logging
import typing

from ase import Atoms
from ase.build import bulk, stack
from ase.data import chemical_symbols

from src.core import AbstractGenerator
from src.domain_models.config import InterfaceTarget, StructureGeneratorConfig


class StructureGenerator(AbstractGenerator):
    """Generates localized candidate structures around an uncertain anchor."""

    def __init__(self, config: StructureGeneratorConfig) -> None:
        self.config = config

    def generate_local_candidates(self, s0: Atoms, n: int = 20) -> typing.Iterator[Atoms]:
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

        return _generator()

    def _validate_interface_elements(self, elements: list[str]) -> None:
        """Helper to ensure we only process chemically valid materials."""
        import re

        for el in elements:
            # First ensure only basic letters are used in naming
            if not re.match(r"^[A-Za-z]+$", el):
                msg = f"Invalid format for element name: {el}"
                raise ValueError(msg)

            # Prevent arbitrary massive strings passing Regex
            if len(el) > 20:
                msg = f"Element name exceeds limits: {el}"
                raise ValueError(msg)

            # If the user passes a compound like "FePt", we break it into symbols
            parsed_symbols = re.findall(r"[A-Z][a-z]*", el)
            for symbol in parsed_symbols:
                if symbol not in chemical_symbols:
                    msg = f"Unsupported or disallowed interface element: {symbol}"
                    raise ValueError(msg)

    def generate_interface(self, target: InterfaceTarget) -> Atoms:
        """Generates an interface structure based on an InterfaceTarget config."""
        # Security: validate elements before passing to ASE
        self._validate_interface_elements([target.element1, target.element2])

        logging.info(
            f"Generating interface between {target.element1} (face {target.face1}) "
            f"and {target.element2} (face {target.face2})."
        )

        try:
            # Basic FePt structure builder
            def _build_fept() -> Atoms:
                # To ensure Fe and Pt are present in a basic bulk representation
                mat = bulk(
                    "Fe", crystalstructure="fcc", a=self.config.fept_lattice_constant, cubic=True
                )  # type: ignore[no-untyped-call]
                # Replace half the atoms with Pt to make it FePt L1_0 like
                for i in range(len(mat)):
                    if i % 2 == 0:
                        mat[i].symbol = "Pt"
                return mat

            # Basic MgO structure builder
            def _build_mgo() -> Atoms:
                return bulk(
                    "MgO",
                    crystalstructure="rocksalt",
                    a=self.config.mgo_lattice_constant,
                    basis=[[0, 0, 0], [0.5, 0.5, 0.5]],
                )  # type: ignore[no-untyped-call]

            builders = {"FePt": _build_fept, "MgO": _build_mgo}

            def _build_generic(element: str) -> Atoms:
                if element in builders:
                    return builders[element]()
                return bulk(element)  # type: ignore[no-untyped-call]

            mat1 = _build_generic(target.element1)
            mat2 = _build_generic(target.element2)

            mat2.set_cell(mat1.get_cell(), scale_atoms=True)  # type: ignore[no-untyped-call]
        except Exception as e:
            logging.exception("Failed to generate interface")
            msg = f"Interface generation failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return stack(mat1, mat2, axis=2, maxstrain=self.config.interface_max_strain)  # type: ignore[no-untyped-call]
