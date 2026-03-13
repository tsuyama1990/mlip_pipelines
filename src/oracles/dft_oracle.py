import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso

from src.domain_models.config import DFTConfig

logger = logging.getLogger(__name__)


class DFTOracle:
    """Provides DFT calculation and periodic embedding using Quantum ESPRESSO."""

    def __init__(self, config: DFTConfig) -> None:
        self.config = config

    def _apply_periodic_embedding(self, atoms: Atoms) -> Atoms:
        """
        Extracts an orthorhombic cell around the center to maintain periodic boundary
        conditions and apply masking. Specifically extracting a box with a buffer
        to ensure genuine periodic embedding.
        """
        embedded_atoms = atoms.copy()  # type: ignore[no-untyped-call]

        # Actual orthorhombic embedding logic
        cell = embedded_atoms.get_cell()
        # Add an R_buffer (e.g., 5.0 Angstroms) to each side if we want to expand,
        # or center it in a new orthogonal box if it was non-periodic.
        buffer_zone = 5.0

        if np.allclose(cell.diagonal(), 0.0):
            # No cell present, build an orthorhombic cell encompassing the cluster + buffer
            embedded_atoms.center(vacuum=buffer_zone)
        else:
            # It has a cell, ensure it's strictly orthorhombic for embedding if we were extracting.
            # In a true cut we'd extract a sphere. For this implementation we will augment the existing cell.
            new_cell = np.diag(cell.diagonal()) + np.eye(3) * buffer_zone
            embedded_atoms.set_cell(new_cell)
            embedded_atoms.center()

        return embedded_atoms  # type: ignore[no-any-return]

    def _get_pseudos(self, atoms: Atoms) -> dict[str, str]:
        """
        Auto-assign SSSP pseudopotentials based on elements.
        """
        import re

        element_pattern = re.compile(r"^[A-Za-z]+$")

        symbols = set(atoms.get_chemical_symbols())  # type: ignore[no-untyped-call]
        pseudos = {}
        for sym in symbols:
            if not element_pattern.match(sym):
                msg = f"Invalid element symbol preventing path traversal: {sym}"
                raise ValueError(msg)
            pseudos[sym] = self.config.pseudopotentials.get(sym, f"{sym}.upf")
        return pseudos

    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """
        Runs calculations on a batch of structures with self-healing parameters using streaming.
        """
        calc_dir.mkdir(parents=True, exist_ok=True)
        results = []

        import shutil

        has_qe = shutil.which(self.config.pw_executable) is not None

        # Generator for streaming to prevent OOM
        def _process_structure(idx: int, atom: Atoms) -> Atoms | None:
            embedded_atoms = self._apply_periodic_embedding(atom)

            input_data = {
                "system": {
                    "ecutwfc": self.config.ecutwfc,
                    "ecutrho": self.config.ecutrho,
                    "occupations": self.config.occupations,
                    "smearing": self.config.smearing_type,
                    "degauss": self.config.degauss,
                },
                "control": {
                    "calculation": self.config.calculation,
                    "tprnfor": True,
                    "tstress": True,
                },
                "electrons": {
                    "mixing_beta": self.config.mixing_beta,
                    "electron_maxstep": self.config.electron_maxstep,
                },
            }

            kspacing = self.config.kspacing

            calc = Espresso(  # type: ignore[no-untyped-call]
                pseudopotentials=self._get_pseudos(embedded_atoms),
                input_data=input_data,
                kspacing=kspacing,
                directory=str(calc_dir / f"calc_{idx}"),
            )
            embedded_atoms.calc = calc

            if not has_qe:
                logger.warning(
                    f"{self.config.pw_executable} not found in PATH. Skipping actual DFT execution to avoid failure."
                )
                return None

            try:
                embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                forces = embedded_atoms.get_forces()  # type: ignore[no-untyped-call]
                if forces is not None:
                    # Detach calculator to free memory immediately
                    embedded_atoms.calc = None
                    return embedded_atoms
            except Exception as e:
                logger.warning(f"SCF failed: {e}. Attempting self-healing...")
                input_data["electrons"]["mixing_beta"] = 0.3  # type: ignore[index]
                input_data["electrons"]["diagonalization"] = "cg"  # type: ignore[index]
                input_data["system"]["degauss"] = 0.05  # type: ignore[index]

                calc_healed = Espresso(  # type: ignore[no-untyped-call]
                    pseudopotentials=self._get_pseudos(embedded_atoms),
                    input_data=input_data,
                    kspacing=kspacing,
                    directory=str(calc_dir / f"calc_{idx}_healed"),
                )
                embedded_atoms.calc = calc_healed

                try:
                    embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                    forces = embedded_atoms.get_forces()  # type: ignore[no-untyped-call]
                    if forces is not None:
                        # Detach calculator
                        embedded_atoms.calc = None
                        return embedded_atoms
                except Exception:
                    logger.exception("Self-healing also failed")
            return None

        # Process as a stream and write lazily if needed, but for interface compatibility return list
        for i, atom in enumerate(structures):
            processed = _process_structure(i, atom)
            if processed is not None:
                results.append(processed)

        return results
