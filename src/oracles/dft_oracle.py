import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.data import atomic_numbers

from src.domain_models.config import OracleConfig
from src.oracles.base import BaseOracle


class DFTManager(BaseOracle):
    """Orchestrates Quantum ESPRESSO via ASE for labeling."""

    def __init__(self, config: OracleConfig) -> None:
        self.config = config

    def _apply_periodic_embedding(self, atoms: Atoms) -> Atoms:
        """Applies Periodic Embedding to create Orthorhombic Box."""
        if not isinstance(atoms, Atoms):
            msg = "Expected ASE Atoms object"
            raise TypeError(msg)

        if len(atoms) == 0:
            msg = "Cannot embed an empty structure."
            raise ValueError(msg)

        embedded: Atoms = atoms.copy()  # type: ignore[no-untyped-call]
        pos = embedded.get_positions()  # type: ignore[no-untyped-call]

        if np.isnan(pos).any() or np.isinf(pos).any():
            msg = "Atomic positions contain NaN or Inf values."
            raise ValueError(msg)

        max_coord = self.config.max_coord
        if np.any(np.abs(pos) > max_coord):
            msg = f"Atomic positions exceed maximum allowed coordinates ({max_coord})"
            raise ValueError(msg)

        if len(atoms) > self.config.max_atoms:
            msg = f"Atomic structure is too large (exceeds {self.config.max_atoms} atoms). Potential memory exhaustion."
            raise ValueError(msg)

        # DoS Protection: validate global atoms
        global_atoms = atoms.get_global_number_of_atoms()  # type: ignore[no-untyped-call]
        if global_atoms > self.config.max_atoms * 10:
            msg = f"Global structure size ({global_atoms}) exceeds extreme physical bounds, possible DoS attempt."
            raise ValueError(msg)

        self._validate_symbols(atoms)
        self._validate_distances(embedded)

        min_pos = pos.min(axis=0)
        max_pos = pos.max(axis=0)

        buffer = self.config.buffer_size
        if buffer < 0:
            msg = "buffer_size must be positive"
            raise ValueError(msg)
        lengths = max_pos - min_pos + 2 * buffer

        if any(self.config.max_cell_dimension < L for L in lengths):
            msg = "Calculated cell dimensions are too large, potential memory exhaustion."
            raise ValueError(msg)

        center = (max_pos + min_pos) / 2
        box_center = lengths / 2
        shifted_pos = pos + box_center - center
        embedded.set_positions(shifted_pos)  # type: ignore[no-untyped-call]

        embedded.set_cell([[lengths[0], 0.0, 0.0], [0.0, lengths[1], 0.0], [0.0, 0.0, lengths[2]]])  # type: ignore[no-untyped-call]
        embedded.set_pbc(True)  # type: ignore[no-untyped-call]
        return embedded

    def _validate_symbols(self, atoms: Atoms) -> None:
        for symbol in atoms.get_chemical_symbols():
            if symbol not in atomic_numbers:
                msg = f"Invalid chemical symbol detected in structure: {symbol}"
                raise ValueError(msg)

    def _validate_distances(self, embedded: Atoms) -> None:
        distances = embedded.get_all_distances()
        np.fill_diagonal(distances, np.inf)
        if np.any(distances < self.config.min_atom_distance):
            msg = f"Structure contains overlapping atoms (distance < {self.config.min_atom_distance} A)."
            raise ValueError(msg)

    def _get_pseudo_dir_path(self) -> Path:
        import os

        try:
            raw_pseudo_dir = Path(self.config.pseudo_dir)
            if not raw_pseudo_dir.is_absolute():
                msg = (
                    f"Pseudopotential directory must be an absolute path: {self.config.pseudo_dir}"
                )
                raise ValueError(msg)
            pseudo_dir_path = raw_pseudo_dir.resolve(strict=True)

            # Directory traversal prevention
            restricted_prefixes = [
                "/etc",
                "/bin",
                "/usr",
                "/sbin",
                "/var",
                "/lib",
                "/boot",
                "/root",
            ]
            for restricted in restricted_prefixes:
                try:
                    is_restricted = (
                        os.path.commonpath([restricted, str(pseudo_dir_path)]) == restricted
                    )
                except ValueError:
                    continue
                if is_restricted:
                    msg = f"Directory cannot be a system directory: {restricted}"
                    raise ValueError(msg)
        except FileNotFoundError as e:
            msg = f"Pseudopotential directory not found: {self.config.pseudo_dir}"
            raise FileNotFoundError(msg) from e
        else:
            if not pseudo_dir_path.is_dir():
                msg = (
                    f"Pseudopotential directory is not a valid directory: {self.config.pseudo_dir}"
                )
                raise ValueError(msg)
            return pseudo_dir_path

    def _raise_upf_error(self, message: str) -> None:
        raise ValueError(message)

    def _check_upf_stat_and_content(self, fd: int, upf_name: str) -> None:
        import os
        import stat

        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            self._raise_upf_error(f"Pseudopotential must be a regular file: {upf_name}")

        if st.st_uid != os.getuid():
            self._raise_upf_error(
                f"Pseudopotential file ownership violation: {upf_name} is not owned by the current user."
            )

        if st.st_size > 10 * 1024 * 1024:  # DoS Protection against massive files
            self._raise_upf_error(f"Pseudopotential file too large (>10MB): {upf_name}")

        with os.fdopen(fd, "r", encoding="utf-8") as f:
            content = f.read()

        if "<UPF" not in content and "PP_INFO" not in content:
            self._raise_upf_error(
                f"Invalid UPF format for pseudopotential: {upf_name}. Missing required opening tags."
            )
        if "</UPF>" not in content and "PP_INFO" not in content:
            self._raise_upf_error(
                f"Invalid UPF format for pseudopotential: {upf_name}. Missing required closing tags."
            )

    def _validate_upf_content(self, upf_path: Path, upf_name: str, pseudo_dir_path: Path) -> None:
        import os

        # Single canonicalization to avoid TOCTOU on path resolution
        canonical_upf = Path(os.path.realpath(str(upf_path)))
        canonical_pseudo_dir = Path(os.path.realpath(str(pseudo_dir_path)))

        if not str(canonical_upf).startswith(str(canonical_pseudo_dir)):
            msg = f"Path traversal detected: {upf_name} resolves outside trusted base."
            raise ValueError(msg)

        try:
            fd = os.open(str(canonical_upf), os.O_RDONLY)
        except OSError as e:
            msg = f"Failed to securely open pseudopotential: {upf_name}"
            raise FileNotFoundError(msg) from e

        try:
            self._check_upf_stat_and_content(fd, upf_name)
        except UnicodeDecodeError as e:
            msg = f"File is corrupted or not utf-8 text: {upf_name}"
            raise ValueError(msg) from e
        except ValueError:
            # Re-raise correctly if _check_upf_stat_and_content threw.
            # Note: os.fdopen() closes fd when context manager exits.
            raise
        except Exception:
            import contextlib

            with contextlib.suppress(OSError):
                os.close(fd)
            raise

    def _validate_pseudopotentials(self, symbols: set[str]) -> dict[str, str]:
        import os

        pseudos = {}
        pseudo_dir_path = self._get_pseudo_dir_path()

        for el in symbols:
            # Domain-level validation relies on ASE's atomic_numbers dictionary
            if el not in atomic_numbers:
                msg = f"Invalid chemical symbol detected: {el}"
                raise ValueError(msg)

            upf_name = f"{el}.upf"
            # Strict resolution before usage to prevent directory escape aliases
            try:
                upf_path = pseudo_dir_path.joinpath(upf_name).resolve(strict=True)
            except FileNotFoundError as e:
                msg = f"Pseudopotential file not found: {upf_name} in {pseudo_dir_path}"
                raise FileNotFoundError(msg) from e

            # Robust pre-resolution TOCTOU validation using commonpath
            try:
                is_safe = os.path.commonpath([str(pseudo_dir_path), str(upf_path)]) == str(
                    pseudo_dir_path
                )
            except ValueError as e:
                msg = f"Invalid path constructed for pseudopotential: {upf_name}"
                raise ValueError(msg) from e

            if not is_safe:
                msg = f"Path traversal detected: {upf_name}"
                raise ValueError(msg)

            self._validate_upf_content(upf_path, upf_name, pseudo_dir_path)

            pseudos[el] = upf_name
        return pseudos

    def _calculate_kpoints(self, atoms: Atoms) -> list[int]:
        cell = atoms.get_cell()  # type: ignore[no-untyped-call]

        if np.isnan(cell).any() or np.isinf(cell).any():
            msg = "Cell dimensions contain NaN or Inf values."
            raise ValueError(msg)

        b = np.linalg.norm(cell, axis=0)

        if any(x <= self.config.min_cell_dimension or np.isnan(x) or np.isinf(x) for x in b):
            msg = "Cell dimensions must be strictly positive and finite for kspacing calculation"
            raise ValueError(msg)

        kpts = []
        for x in b:
            # Safe integer arithmetic and bounds checking before appending
            val_float = np.ceil(2 * np.pi / (self.config.kspacing * x))
            if val_float > self.config.max_kpoints:
                msg = f"Calculated dimension k-points ({val_float}) exceeds maximum absolute points ({self.config.max_kpoints})"
                raise ValueError(msg)
            kpts.append(int(val_float))

        # Check total grid size
        if np.prod(kpts) > self.config.max_kpoints:
            msg = f"Calculated k-point grid {kpts} exceeds maximum allowed points ({self.config.max_kpoints}) to prevent computational exhaustion."
            raise ValueError(msg)
        return kpts

    def _get_calculator(self, atoms: Atoms, work_dir: Path) -> Espresso:
        """Creates the ESPRESSO calculator with self-healing parameters."""
        import os
        import tempfile

        # Security: Canonicalize work_dir and check its boundaries
        canonical_work_dir = work_dir.resolve(strict=True)
        is_safe = False
        allowed_bases = [
            str(Path(tempfile.gettempdir()).resolve(strict=True)),
        ]
        if hasattr(self.config, "project_root"):
            allowed_bases.append(str(Path(self.config.project_root).resolve(strict=True)))

        try:
            st = os.lstat(canonical_work_dir)
            for safe_base in allowed_bases:
                base_st = os.lstat(safe_base)
                try:
                    if (
                        os.path.commonpath([safe_base, str(canonical_work_dir)]) == safe_base
                        and st.st_dev == base_st.st_dev
                    ):  # Additional mount point check
                        is_safe = True
                        break
                except ValueError:
                    pass
        except OSError:
            pass

        if not is_safe:
            msg = f"Work directory {canonical_work_dir} is not within trusted boundaries or device mismatch."
            raise ValueError(msg)

        symbols: set[str] = set(atoms.get_chemical_symbols())  # type: ignore[no-untyped-call]

        try:
            pseudos = self._validate_pseudopotentials(symbols)
        except Exception as e:
            msg = f"Failed to validate pseudopotentials for symbols {symbols}. Ensure valid UPF files exist in {self.config.pseudo_dir}. Error: {e}"
            raise ValueError(msg) from e

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

    def _update_calc_parameters(self, calc: Espresso, attempt: int) -> None:
        """Securely mutates calculator parameters for self-healing."""
        if attempt == 0:
            calc.parameters["input_data"]["electrons"]["mixing_beta"] = 0.3
        elif attempt == 1:
            calc.parameters["input_data"]["electrons"]["diagonalization"] = "cg"
        else:
            calc.parameters["input_data"]["electrons"]["mixing_beta"] = 0.1

    def compute_batch(self, structures: list[Atoms], calc_dir: Path) -> list[Atoms]:
        """Runs DFT on a batch of structures with self-healing."""
        import os
        import tempfile

        from src.core.exceptions import OracleConvergenceError

        # Secure workspace execution: ensure calc_dir is inside tempfile.gettempdir()
        tmp_base = Path(tempfile.gettempdir()).resolve(strict=True)
        resolved_calc_dir = calc_dir.resolve(strict=False)

        # TOCTOU symlink protection: ensure the resolved absolute path starts strictly
        # from tmp_base, mitigating symlink aliases before mkdir
        resolved_str = os.path.realpath(str(resolved_calc_dir))
        tmp_base_str = os.path.realpath(str(tmp_base))
        if os.path.commonpath([tmp_base_str, resolved_str]) != tmp_base_str:
            msg = f"Security Violation: Oracle compute batches must be executed inside a trusted temporary directory, not {calc_dir}"
            raise ValueError(msg)

        resolved_calc_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, atoms in enumerate(structures):
            work_dir = resolved_calc_dir / f"struct_{i}"
            work_dir.mkdir(parents=True, exist_ok=True)

            embedded_atoms = self._apply_periodic_embedding(atoms)
            calc = self._get_calculator(embedded_atoms, work_dir)
            embedded_atoms.calc = calc

            success = False
            last_exception = None

            for attempt in range(self.config.max_retries + 1):
                try:
                    embedded_atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                    embedded_atoms.get_forces()  # type: ignore[no-untyped-call]
                    embedded_atoms.get_stress()  # type: ignore[no-untyped-call]
                    results.append(embedded_atoms)
                    success = True
                    break
                except Exception as e:
                    last_exception = e
                    if attempt < self.config.max_retries:
                        logging.warning(
                            f"SCF failed for struct {i} attempt {attempt + 1}: {e}. Attempting self-healing..."
                        )
                        self._update_calc_parameters(calc, attempt)
                    else:
                        logging.warning(
                            f"Failed completely for struct {i} after {self.config.max_retries} retries: {e}"
                        )

            if not success:
                msg = f"Failed to converge structure {i} after {self.config.max_retries} retries."
                raise OracleConvergenceError(msg) from last_exception

        return results
