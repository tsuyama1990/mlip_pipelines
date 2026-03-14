import logging
from pathlib import Path

from ase import Atoms
from ase.calculators.espresso import Espresso

from src.core import AbstractOracle
from src.domain_models.config import OracleConfig


class DFTManager(AbstractOracle):
    """Orchestrates Quantum ESPRESSO via ASE for labeling."""

    def __init__(self, config: OracleConfig) -> None:
        self.config = config

    def _apply_periodic_embedding(self, atoms: Atoms) -> Atoms:
        """Applies Periodic Embedding to create Orthorhombic Box."""
        if len(atoms) == 0:
            msg = "Cannot embed an empty structure."
            raise ValueError(msg)

        # Spec says to define a cubic or orthorhombic box around atoms.
        embedded: Atoms = atoms.copy()  # type: ignore[no-untyped-call]

        # We need an orthorhombic cell for embedding.
        # Find bounds and pad by buffer
        pos = embedded.get_positions()  # type: ignore[no-untyped-call]

        import numpy as np

        if np.isnan(pos).any() or np.isinf(pos).any():
            msg = "Atomic positions contain NaN or Inf values."
            raise ValueError(msg)

        min_pos = pos.min(axis=0)
        max_pos = pos.max(axis=0)

        buffer = self.config.buffer_size
        if buffer < 0:
            msg = "buffer_size must be positive"
            raise ValueError(msg)
        lengths = max_pos - min_pos + 2 * buffer

        # Validate reasonable cell size to prevent memory exhaustion (buffer overflow defense)
        if any(L > 100.0 for L in lengths):
            msg = "Calculated cell dimensions are too large, potential memory exhaustion."
            raise ValueError(msg)

        # Shift positions so they center within the box
        center = (max_pos + min_pos) / 2
        box_center = lengths / 2

        # update positions correctly
        shifted_pos = pos + box_center - center
        embedded.set_positions(shifted_pos)  # type: ignore[no-untyped-call]

        # Set Orthorhombic cell
        embedded.set_cell([[lengths[0], 0.0, 0.0], [0.0, lengths[1], 0.0], [0.0, 0.0, lengths[2]]])  # type: ignore[no-untyped-call]
        embedded.set_pbc(True)  # type: ignore[no-untyped-call]
        return embedded

    def _get_calculator(self, atoms: Atoms, work_dir: Path) -> Espresso:  # noqa: C901
        """Creates the ESPRESSO calculator with self-healing parameters."""
        # Determine pseudopotentials from elements
        import re

        from ase.data import atomic_numbers

        symbols: set[str] = set(atoms.get_chemical_symbols())  # type: ignore[no-untyped-call]
        pseudos = {}

        # Use strict resolution to ensure the base directory exists and is canonical
        try:
            pseudo_dir_path = Path(self.config.pseudo_dir).resolve(strict=True)
            if not pseudo_dir_path.is_absolute() or not pseudo_dir_path.is_dir():
                msg = f"Pseudopotential directory must be an absolute path to a valid directory: {self.config.pseudo_dir}"
                raise ValueError(msg)
        except FileNotFoundError as e:
            msg = f"Pseudopotential directory not found: {self.config.pseudo_dir}"
            raise FileNotFoundError(msg) from e

        for el in symbols:
            # Validate element names strictly against a whitelist of valid chemical symbols structure
            # to prevent path traversal attacks (e.g. element name "../malicious")
            if not re.match(r"^[A-Z][a-z]?$", el):
                msg = "Invalid element name"
                raise ValueError(msg)
            if el not in atomic_numbers:
                msg = f"Invalid chemical symbol detected: {el}"
                raise ValueError(msg)

            upf_name = f"{el}.upf"

            # Additional layer of security: Ensure the resolved path strictly resides within the intended directory
            upf_path = (pseudo_dir_path / upf_name).resolve()

            if not upf_path.is_relative_to(pseudo_dir_path):
                msg = f"Path traversal detected for pseudopotential: {upf_name}"
                raise ValueError(msg)

            if not upf_path.exists():
                msg = f"Pseudopotential file not found: {upf_name} in {pseudo_dir_path}"
                raise FileNotFoundError(msg)
            pseudos[el] = upf_name

        # Check for transition metals
        transition_metals = set(self.config.transition_metals)
        has_tm = any(el in transition_metals for el in symbols)

        # K-points from Kspacing
        cell = atoms.get_cell()  # type: ignore[no-untyped-call]
        import numpy as np

        b = np.linalg.norm(cell, axis=0)  # roughly real lattice vectors lengths

        # Validate cell dimensions for kspacing
        if any(x <= 1e-5 or np.isnan(x) or np.isinf(x) for x in b):
            msg = "Cell dimensions must be strictly positive and finite for kspacing calculation"
            raise ValueError(msg)

        kpts = [int(np.ceil(2 * np.pi / (self.config.kspacing * x))) for x in b]

        # Validated smearing
        degauss = self.config.smearing_width if self.config.smearing_width > 0.0 else 0.02

        input_data = {
            "control": {"calculation": self.config.calculation},
            "system": {
                "ecutwfc": self.config.ecutwfc,
                "ecutrho": self.config.ecutrho,
                "occupations": self.config.occupations,
                "smearing": self.config.smearing,
                "degauss": degauss,
            },
            "electrons": {
                "mixing_beta": self.config.mixing_beta,
                "diagonalization": self.config.diagonalization,
            },
        }

        if has_tm:
            input_data["system"]["nspin"] = 2  # type: ignore[index]

            # Start magnetisation heuristics
            start_mag = {}
            for _i, el in enumerate(atoms.get_chemical_symbols()):  # type: ignore[no-untyped-call]
                if el in transition_metals:
                    start_mag[el] = 1.0  # High spin initialization

            input_data["system"]["starting_magnetization"] = start_mag  # type: ignore[index]

        return Espresso(  # type: ignore[no-untyped-call]
            pseudopotentials=pseudos,
            pseudo_dir=self.config.pseudo_dir,
            tstress=True,
            tprnfor=True,
            kpts=kpts,
            directory=str(work_dir),
            input_data=input_data,
        )

    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """Runs DFT on a batch of structures with self-healing."""
        calc_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, atoms in enumerate(structures):
            work_dir = calc_dir / f"struct_{i}"
            work_dir.mkdir(parents=True, exist_ok=True)

            embedded_atoms = self._apply_periodic_embedding(atoms)
            calc = self._get_calculator(embedded_atoms, work_dir)
            embedded_atoms.calc = calc

            try:
                # Calculate properties.
                # Since ASE just executes pw.x, we assume pw.x is in PATH.
                # If not, ASE will fail. We need to handle this robustly.
                embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                embedded_atoms.get_forces()  # type: ignore[no-untyped-call]
                embedded_atoms.get_stress()  # type: ignore[no-untyped-call]

                results.append(embedded_atoms)

            except Exception as e:
                # Self-healing logic for SCF convergence error:
                logging.warning(f"SCF failed for struct {i}: {e}. Attempting self-healing...")

                # Retry 1: Lower mixing beta
                calc.parameters["input_data"]["electrons"]["mixing_beta"] = 0.3
                try:
                    embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                    results.append(embedded_atoms)
                    continue
                except Exception as inner_e:
                    logging.warning(f"Self-healing retry 1 failed: {inner_e}")

                # Retry 2: Change diagonalization
                calc.parameters["input_data"]["electrons"]["diagonalization"] = "cg"
                try:
                    embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                    results.append(embedded_atoms)
                except Exception as final_e:
                    logging.warning(f"Failed completely for struct {i}: {final_e}")

        return results
