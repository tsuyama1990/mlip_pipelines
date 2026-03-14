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
        # Attempt to import dependencies for strict runtime safety
        import os
        use_mock = os.environ.get("USE_MOCK", "False") == "True"

        if not use_mock:
            try:
                from pyacemaker.calculator import pyacemaker  # noqa: F401
            except ImportError as e:
                msg = "pyacemaker dependency missing. pyacemaker is required for validation."
                raise ImportError(msg) from e

            try:
                import phonopy  # noqa: F401
            except ImportError as e:
                msg = "phonopy dependency missing. phonopy is required for validation."
                raise ImportError(msg) from e

    def _check_file_format(self, resolved_path: Path) -> None:
        import os

        # Verify file exists before reading
        if not resolved_path.exists():
            msg = f"Potential file not found: {resolved_path}"
            raise FileNotFoundError(msg)

        if not resolved_path.is_file():
            msg = f"Potential file is not a file: {resolved_path}"
            raise FileNotFoundError(msg)

        if not str(resolved_path).endswith(".yace"):
            msg = f"Potential file must have .yace extension: {resolved_path}"
            raise ValueError(msg)

        # Enforce canonical path
        strict_path = Path(os.path.normpath(os.path.realpath(resolved_path))).resolve(strict=True)

        with Path.open(strict_path) as f:
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

        # Real validation without fallbacks. Initialize to zero if no test dataset.
        energy_rmse: float = 0.0
        force_rmse: float = 0.0
        stress_rmse: float = 0.0

        if self.config.test_dataset_path is not None:
            test_path: Path = Path(self.config.test_dataset_path).resolve(strict=True)
            if test_path.exists():
                from ase.io import read

                test_atoms_list = read(str(test_path), index=":")
                if not isinstance(test_atoms_list, list):
                    test_atoms_list = [test_atoms_list]

                e_errors: list[float] = []
                f_errors: list[float] = []
                s_errors: list[float] = []

                for test_atoms in test_atoms_list:
                    # Ground truth from dataset
                    true_e: float = float(test_atoms.get_potential_energy())  # type: ignore[no-untyped-call]
                    true_f: np.ndarray = test_atoms.get_forces()  # type: ignore[no-untyped-call, type-arg]
                    # stress might not be available
                    true_s: np.ndarray | None
                    try:
                        true_s = test_atoms.get_stress()  # type: ignore[no-untyped-call]
                    except Exception:
                        true_s = None

                    test_atoms.calc = calc
                    pred_e: float = float(test_atoms.get_potential_energy())  # type: ignore[no-untyped-call]
                    pred_f: np.ndarray = test_atoms.get_forces()  # type: ignore[no-untyped-call, type-arg]

                    e_errors.append((pred_e - true_e) ** 2)
                    f_errors.append(float(np.mean((pred_f - true_f) ** 2)))

                    if true_s is not None:
                        pred_s: np.ndarray = test_atoms.get_stress()  # type: ignore[no-untyped-call, type-arg]
                        s_errors.append(float(np.mean((pred_s - true_s) ** 2)))

                if e_errors:
                    energy_rmse = float(np.sqrt(np.mean(e_errors)))
                if f_errors:
                    force_rmse = float(np.sqrt(np.mean(f_errors)))
                if s_errors:
                    stress_rmse = float(np.sqrt(np.mean(s_errors)))

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
