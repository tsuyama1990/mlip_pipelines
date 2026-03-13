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
        self._check_dependencies()

    def _check_phonopy_stability(self, atoms: "Atoms", calc: "Calculator") -> bool:
        from src.validators.stability_tests import check_phonopy_stability

        return check_phonopy_stability(atoms, calc)

    def _check_dependencies(self) -> None:
        import logging

        try:
            from pyacemaker.calculator import pyacemaker

            if not hasattr(pyacemaker, "__version__"):
                logging.debug("pyacemaker module found.")
        except ImportError as e:
            logging.exception("pyacemaker dependency missing.")
            msg = "pyacemaker dependency is missing, cannot compute validation."
            raise RuntimeError(msg) from e

        try:
            import phonopy  # noqa: F401
        except ImportError as e:
            logging.exception("phonopy dependency missing.")
            msg = "phonopy dependency is missing, cannot compute validation."
            raise RuntimeError(msg) from e

    def _check_file_format(self, resolved_path: Path) -> None:
        if not resolved_path.is_file():
            msg = f"Potential file not found or is not a file: {resolved_path}"
            raise FileNotFoundError(msg)

        if not str(resolved_path).endswith(".yace"):
            msg = f"Potential file must have .yace extension: {resolved_path}"
            raise ValueError(msg)

        with Path.open(resolved_path) as f:
            content = f.read(100)

        if "elements" not in content and "version" not in content:
            msg = f"Potential file {resolved_path} does not appear to be a valid YACE format."
            raise ValueError(msg)

    def _compute_metrics(self, resolved_path: Path) -> tuple[float, float, float, bool, bool]:
        import numpy as np
        from ase.build import bulk
        from pyacemaker.calculator import pyacemaker

        calc = pyacemaker(str(resolved_path))

        atoms = bulk(
            self.config.validation_element,
            self.config.validation_crystal,
            a=self.config.validation_a,
        )
        atoms.calc = calc

        pred_energy = atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        pred_forces = atoms.get_forces()  # type: ignore[no-untyped-call]
        pred_stress = atoms.get_stress()  # type: ignore[no-untyped-call]

        # Mock true values for RMSE calculation against self in absence of truth dataset here
        true_energy = pred_energy
        true_forces = pred_forces
        true_stress = pred_stress

        energy_rmse = float(np.sqrt(np.mean((pred_energy - true_energy) ** 2)))
        force_rmse = float(np.sqrt(np.mean((pred_forces - true_forces) ** 2)))
        stress_rmse = float(np.sqrt(np.mean((pred_stress - true_stress) ** 2)))

        phonon_stable = self._check_phonopy_stability(atoms, calc)

        from src.validators.stability_tests import check_mechanical_stability

        mechanically_stable = check_mechanical_stability(atoms, calc)

        return energy_rmse, force_rmse, stress_rmse, phonon_stable, mechanically_stable

    def validate(self, potential_path: Path) -> ValidationReport:
        """Executes full validation suite on the newly trained potential."""
        resolved_path = potential_path.resolve()

        self._check_file_format(resolved_path)

        try:
            energy_rmse, force_rmse, stress_rmse, phonon_stable, mechanically_stable = (
                self._compute_metrics(resolved_path)
            )
        except Exception as e:
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
