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

        embedded: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
        pos = embedded.get_positions()  # type: ignore[no-untyped-call]

        import numpy as np

        if np.isnan(pos).any() or np.isinf(pos).any():
            msg = "Atomic positions contain NaN or Inf values."
            raise ValueError(msg)

        max_coord = 1e5
        if np.any(np.abs(pos) > max_coord):
            msg = f"Atomic positions exceed maximum allowed coordinates ({max_coord})"
            raise ValueError(msg)

        if len(atoms) > 10000:
            msg = (
                "Atomic structure is too large (exceeds 10000 atoms). Potential memory exhaustion."
            )
            raise ValueError(msg)

        from ase.data import atomic_numbers

        for symbol in atoms.get_chemical_symbols():
            if symbol not in atomic_numbers:
                msg = f"Invalid chemical symbol detected in structure: {symbol}"
                raise ValueError(msg)

        # Check for overlapping atoms (distance < 0.1 A)
        distances = embedded.get_all_distances()
        np.fill_diagonal(distances, np.inf)
        if np.any(distances < 0.1):
            msg = "Structure contains overlapping atoms (distance < 0.1 A)."
            raise ValueError(msg)

        min_pos = pos.min(axis=0)
        max_pos = pos.max(axis=0)

        buffer = self.config.buffer_size
        if buffer < 0:
            msg = "buffer_size must be positive"
            raise ValueError(msg)
        lengths = max_pos - min_pos + 2 * buffer

        if any(L > 1000.0 for L in lengths):
            msg = "Calculated cell dimensions are too large, potential memory exhaustion."
            raise ValueError(msg)

        center = (max_pos + min_pos) / 2
        box_center = lengths / 2
        shifted_pos = pos + box_center - center
        embedded.set_positions(shifted_pos)  # type: ignore[no-untyped-call]

        embedded.set_cell([[lengths[0], 0.0, 0.0], [0.0, lengths[1], 0.0], [0.0, 0.0, lengths[2]]])  # type: ignore[no-untyped-call]
        embedded.set_pbc(True)  # type: ignore[no-untyped-call]
        return embedded

    def _validate_pseudopotentials(self, symbols: set[str]) -> dict[str, str]:
        import re

        from ase.data import atomic_numbers

        pseudos = {}
        try:
            raw_pseudo_dir = Path(self.config.pseudo_dir)
            if not raw_pseudo_dir.is_absolute():
                msg = (
                    f"Pseudopotential directory must be an absolute path: {self.config.pseudo_dir}"
                )
                raise ValueError(msg)
            pseudo_dir_path = raw_pseudo_dir.resolve(strict=True)
            if not pseudo_dir_path.is_dir():
                msg = (
                    f"Pseudopotential directory is not a valid directory: {self.config.pseudo_dir}"
                )
                raise ValueError(msg)
        except FileNotFoundError as e:
            msg = f"Pseudopotential directory not found: {self.config.pseudo_dir}"
            raise FileNotFoundError(msg) from e

        for el in symbols:
            if not re.match(r"^[A-Z][a-z]?$", el):
                msg = "Invalid element name"
                raise ValueError(msg)
            if el not in atomic_numbers:
                msg = f"Invalid chemical symbol detected: {el}"
                raise ValueError(msg)
            upf_name = f"{el}.upf"
            upf_path = pseudo_dir_path / upf_name
            if not upf_path.exists():
                msg = f"Pseudopotential file not found: {upf_name} in {pseudo_dir_path}"
                raise FileNotFoundError(msg)
            resolved_upf_path = upf_path.resolve(strict=True)
            if not resolved_upf_path.is_relative_to(pseudo_dir_path.resolve(strict=True)):
                msg = f"Strict path traversal detected for pseudopotential: {upf_name}"
                raise ValueError(msg)
            pseudos[el] = upf_name
        return pseudos

    def _calculate_kpoints(self, atoms: Atoms) -> list[int]:
        import numpy as np

        cell = atoms.get_cell()  # type: ignore[no-untyped-call]
        b = np.linalg.norm(cell, axis=0)

        if any(x <= 1e-5 or np.isnan(x) or np.isinf(x) for x in b):
            msg = "Cell dimensions must be strictly positive and finite for kspacing calculation"
            raise ValueError(msg)

        kpts = [int(np.ceil(2 * np.pi / (self.config.kspacing * x))) for x in b]
        if np.prod(kpts) > 1000:
            msg = f"Calculated k-point grid {kpts} exceeds maximum allowed points (1000) to prevent computational exhaustion."
            raise ValueError(msg)
        return kpts

    def _get_calculator(self, atoms: Atoms, work_dir: Path) -> Espresso:
        """Creates the ESPRESSO calculator with self-healing parameters."""
        symbols: set[str] = set(atoms.get_chemical_symbols())  # type: ignore[no-untyped-call]
        pseudos = self._validate_pseudopotentials(symbols)
        kpts = self._calculate_kpoints(atoms)

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

        transition_metals = set(self.config.transition_metals)
        if any(el in transition_metals for el in symbols):
            input_data["system"]["nspin"] = 2  # type: ignore[index]
            start_mag = {el: 1.0 for el in atoms.get_chemical_symbols() if el in transition_metals}  # type: ignore[no-untyped-call]
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
                embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                embedded_atoms.get_forces()  # type: ignore[no-untyped-call]
                embedded_atoms.get_stress()  # type: ignore[no-untyped-call]
                results.append(embedded_atoms)
            except Exception as e:
                logging.warning(f"SCF failed for struct {i}: {e}. Attempting self-healing...")
                calc.parameters["input_data"]["electrons"]["mixing_beta"] = 0.3
                try:
                    embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                    results.append(embedded_atoms)
                    continue
                except Exception as inner_e:
                    logging.warning(f"Self-healing retry 1 failed: {inner_e}")

                calc.parameters["input_data"]["electrons"]["diagonalization"] = "cg"
                try:
                    embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                    results.append(embedded_atoms)
                except Exception as final_e:
                    logging.warning(f"Failed completely for struct {i}: {final_e}")

        return results
