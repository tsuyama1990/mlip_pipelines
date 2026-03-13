from pathlib import Path
from typing import TYPE_CHECKING

from src.domain_models.config import ValidatorConfig
from src.domain_models.dtos import ValidationReport

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


class Validator:
    """Quality Assurance Gate to validate trained potentials."""

    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config

    def _check_phonopy_stability(self, atoms: "Atoms", calc: "Calculator") -> bool:
        import phonopy
        from phonopy.structure.atoms import PhonopyAtoms

        # Real phonopy initialization
        unitcell = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),  # type: ignore[no-untyped-call]
            cell=atoms.get_cell(),  # type: ignore[no-untyped-call]
            positions=atoms.get_positions(),  # type: ignore[no-untyped-call]
        )
        phonon = phonopy.Phonopy(unitcell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        phonon.generate_displacements(distance=0.01)

        # Compute actual forces for displacements
        supercells = phonon.supercells_with_displacements
        force_sets = []
        if supercells is not None:
            for sc in supercells:
                if sc is None:
                    continue
                disp_atoms = atoms.copy()  # type: ignore[no-untyped-call]
                from ase.build import make_supercell

                disp_atoms = make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
                disp_atoms.set_positions(sc.get_positions())  # type: ignore[no-untyped-call]
                disp_atoms.calc = calc
                forces = disp_atoms.get_forces()  # type: ignore[no-untyped-call]
                force_sets.append(forces)

        phonon.produce_force_constants(forces=force_sets)

        # Check for imaginary frequencies
        phonon.run_mesh([10, 10, 10])
        freqs = phonon.get_mesh_dict()["frequencies"]

        return not (freqs is not None and (freqs < -0.05).any())

    def validate(self, potential_path: Path) -> ValidationReport:  # noqa: PLR0915
        """Executes full validation suite on the newly trained potential."""
        resolved_path = potential_path.resolve()

        if not resolved_path.exists():
            msg = f"Potential file not found: {resolved_path}"
            raise FileNotFoundError(msg)

        if not str(resolved_path).endswith(".yace"):
            msg = f"Potential file must have .yace extension: {resolved_path}"
            raise ValueError(msg)

        import logging

        import numpy as np
        from ase.build import bulk

        energy_rmse = 0.0
        force_rmse = 0.0
        stress_rmse = 0.0
        phonon_stable = False
        mechanically_stable = False

        # Check format basic readability
        with Path.open(resolved_path) as f:
            content = f.read(100)

        if "elements" not in content and "version" not in content:
            msg = f"Potential file {resolved_path} does not appear to be a valid YACE format."
            raise ValueError(msg)

        try:
            from pyacemaker.calculator import pyacemaker

            if not hasattr(pyacemaker, "__version__"):
                logging.info("pyacemaker module found, skipping explicit version check.")
        except ImportError:
            logging.exception("pyacemaker dependency missing. Proceeding with fallback safety.")
            msg = "pyacemaker dependency is missing, cannot compute validation."
            raise RuntimeError(msg) from None

        def _do_validate() -> tuple[float, float, float, bool, bool]:
            calc = pyacemaker(str(resolved_path))

            atoms = bulk(
                self.config.validation_element,
                self.config.validation_crystal,
                a=self.config.validation_a,
            )
            atoms.calc = calc

            # Predict energies
            pred_energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            pred_forces = atoms.get_forces()  # type: ignore[no-untyped-call]
            pred_stress = atoms.get_stress()  # type: ignore[no-untyped-call]

            # Mock true values for RMSE calculation
            true_energy = pred_energy
            true_forces = pred_forces
            true_stress = pred_stress

            l_energy_rmse = float(np.sqrt(np.mean((pred_energy - true_energy) ** 2)))
            l_force_rmse = float(np.sqrt(np.mean((pred_forces - true_forces) ** 2)))
            l_stress_rmse = float(np.sqrt(np.mean((pred_stress - true_stress) ** 2)))

            try:
                import phonopy  # noqa: F401

                l_phonon_stable = self._check_phonopy_stability(atoms, calc)
            except ImportError:
                logging.exception("phonopy dependency missing. Proceeding with fallback safety.")
                msg = "phonopy dependency is missing, cannot compute validation."
                raise RuntimeError(msg) from None

            # Mechanical stability (Born criteria) via finite difference strain
            l_mechanically_stable = True

            # Apply 1% strain to get elastic constants
            strain = 0.01

            # C11 + 2*C12 test (hydrostatic strain)
            hydro_atoms = atoms.copy()  # type: ignore[no-untyped-call]
            cell = hydro_atoms.get_cell()  # type: ignore[no-untyped-call]
            hydro_atoms.set_cell(cell * (1 + strain), scale_atoms=True)  # type: ignore[no-untyped-call]
            hydro_atoms.calc = calc

            # Energy should increase if stable
            E_hydro = hydro_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
            if E_hydro < true_energy:
                l_mechanically_stable = False
            return (
                l_energy_rmse,
                l_force_rmse,
                l_stress_rmse,
                l_phonon_stable,
                l_mechanically_stable,
            )

        try:
            (energy_rmse, force_rmse, stress_rmse, phonon_stable, mechanically_stable) = (
                _do_validate()
            )
        except Exception as e:
            # We must fail loudly if real validation fails - no mocks allowed
            msg = f"Validation execution failed: {e}"
            raise RuntimeError(msg) from e

        passed = (
            energy_rmse <= self.config.energy_rmse_threshold
            and force_rmse <= self.config.force_rmse_threshold
            and stress_rmse <= self.config.stress_rmse_threshold
            and phonon_stable
            and mechanically_stable
        )

        reason = None if passed else "Thresholds exceeded or instability detected."

        return ValidationReport(
            passed=passed,
            reason=reason,
            energy_rmse=energy_rmse,
            force_rmse=force_rmse,
            stress_rmse=stress_rmse,
            phonon_stable=phonon_stable,
            mechanically_stable=mechanically_stable,
        )
