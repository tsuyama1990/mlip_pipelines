import logging
from pathlib import Path

from ase import Atoms
from ase.calculators.espresso import Espresso

from src.domain_models.config import OracleConfig


class DFTManager:
    """Orchestrates Quantum ESPRESSO via ASE for labeling."""

    def __init__(self, config: OracleConfig) -> None:
        self.config = config

    def _apply_periodic_embedding(self, atoms: Atoms) -> Atoms:
        """Applies Periodic Embedding to create Orthorhombic Box."""
        # Spec says to define a cubic or orthorhombic box around atoms.
        embedded = atoms.copy()  # type: ignore[no-untyped-call]

        # We need an orthorhombic cell for embedding.
        # Find bounds and pad by buffer
        pos = embedded.positions
        min_pos = pos.min(axis=0)
        max_pos = pos.max(axis=0)

        buffer = self.config.buffer_size
        lengths = max_pos - min_pos + 2 * buffer

        # Shift positions so they center within the box
        center = (max_pos + min_pos) / 2
        box_center = lengths / 2
        embedded.positions += (box_center - center)

        # Set Orthorhombic cell
        embedded.set_cell([
            [lengths[0], 0.0, 0.0],
            [0.0, lengths[1], 0.0],
            [0.0, 0.0, lengths[2]]
        ])
        embedded.set_pbc(True)
        return embedded

    def _get_calculator(self, atoms: Atoms, work_dir: Path) -> Espresso:
        """Creates the ESPRESSO calculator with self-healing parameters."""
        # Determine pseudopotentials from elements
        symbols = set(atoms.get_chemical_symbols())
        pseudos = {el: f"{el}.upf" for el in symbols}

        # Check for transition metals
        transition_metals = set(self.config.transition_metals)
        has_tm = any(el in transition_metals for el in symbols)

        # K-points from Kspacing
        cell = atoms.get_cell()
        import numpy as np
        b = np.linalg.norm(cell, axis=0) # roughly real lattice vectors lengths
        kpts = [int(np.ceil(2 * np.pi / (self.config.kspacing * x))) if x > 0 else 1 for x in b]

        # Validated smearing
        degauss = self.config.smearing_width if self.config.smearing_width > 0.0 else 0.02

        input_data = {
            "control": {
                "calculation": self.config.calculation
            },
            "system": {
                "ecutwfc": self.config.ecutwfc,
                "ecutrho": self.config.ecutrho,
                "occupations": self.config.occupations,
                "smearing": self.config.smearing,
                "degauss": degauss
            },
            "electrons": {
                "mixing_beta": self.config.mixing_beta,
                "diagonalization": self.config.diagonalization
            }
        }

        if has_tm:
            input_data["system"]["nspin"] = 2

            # Start magnetisation heuristics
            start_mag = {}
            for _i, el in enumerate(atoms.get_chemical_symbols()):
                if el in transition_metals:
                    start_mag[el] = 1.0 # High spin initialization

            input_data["system"]["starting_magnetization"] = start_mag

        return Espresso(
            pseudopotentials=pseudos,
            pseudo_dir=self.config.pseudo_dir,
            tstress=True,
            tprnfor=True,
            kpts=kpts,
            directory=str(work_dir),
            input_data=input_data
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
                embedded_atoms.get_potential_energy()
                embedded_atoms.get_forces()
                embedded_atoms.get_stress()

                results.append(embedded_atoms)

            except Exception as e:
                # Self-healing logic for SCF convergence error:
                logging.warning(f"SCF failed for struct {i}: {e}. Attempting self-healing...")

                # Retry 1: Lower mixing beta
                calc.parameters["input_data"]["electrons"]["mixing_beta"] = 0.3
                try:
                    embedded_atoms.get_potential_energy()
                    results.append(embedded_atoms)
                    continue
                except Exception as inner_e:
                    logging.warning(f"Self-healing retry 1 failed: {inner_e}")

                # Retry 2: Change diagonalization
                calc.parameters["input_data"]["electrons"]["diagonalization"] = "cg"
                try:
                    embedded_atoms.get_potential_energy()
                    results.append(embedded_atoms)
                except Exception as final_e:
                    logging.exception(f"Failed completely for struct {i}: {final_e}")

        return results
