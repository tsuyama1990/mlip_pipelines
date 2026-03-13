import logging
import subprocess
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read

from src.domain_models.config import DynamicsConfig, SystemConfig


class MDInterface:
    """Manages LAMMPS execution using the python module or subprocess."""

    def __init__(self, config: DynamicsConfig, system_config: SystemConfig, project_root: Path) -> None:
        self.config = config
        self.system_config = system_config
        self.project_root = project_root

    def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:  # noqa: C901, PLR0915, PLR0912
        """Runs LAMMPS exploration and monitors for high uncertainty."""
        work_dir.mkdir(parents=True, exist_ok=True)
        dump_file = work_dir / "dump.lammps"

        if potential is not None and not potential.exists():
            msg = f"Potential file not found: {potential}"
            raise FileNotFoundError(msg)

        in_file = work_dir / "in.lammps"

        def get_zbl_mapping(elements: list[str]) -> str:
            return " ".join(str(atomic_numbers.get(el, 1)) for el in elements)

        zbl_elements = get_zbl_mapping(self.system_config.elements)

        import os
        import shutil
        import tempfile

        fd, tmp_path = tempfile.mkstemp(dir=work_dir, text=True)
        try:
            with os.fdopen(fd, "w") as tmp_in_file:
                if potential is None:
                    import re
                    dump_name = dump_file.name
                    if not re.match(r"^[-a-zA-Z0-9_.]+$", dump_name):
                        msg = "Dump file name contains invalid characters"
                        raise ValueError(msg)

                    # Cold start logic: Run MD using only the ZBL baseline potential
                    tmp_in_file.write(f"""
units metal
boundary p p p
atom_style atomic

# Cold start: using only ZBL
pair_style zbl 1.0 2.0
pair_coeff * * {zbl_elements}

# Force dump to extract structures for initial training
dump 1 all custom 10 {dump_name} id type x y z
run {min(self.config.md_steps, 1000)}  # Short run for cold start
""")
                else:
                    # Validate potential path ends with .yace and safely quote it
                    resolved_pot = potential.resolve(strict=True)
                    if not resolved_pot.is_relative_to(self.project_root.resolve()):
                        msg = "Potential path is outside project root"
                        raise ValueError(msg)

                    pot_path_str = str(resolved_pot)
                    if not pot_path_str.endswith(".yace"):
                        msg = "Potential path must end with .yace"
                        raise ValueError(msg)

                    import re
                    # Validate path safe for lammps without shlex quotes to avoid LAMMPS syntax errors
                    # using strict whitelist
                    if not re.match(r"^[-a-zA-Z0-9_./]+$", pot_path_str):
                        msg = "Potential path contains invalid characters for LAMMPS"
                        raise ValueError(msg)

                    dump_name = dump_file.name
                    if not re.match(r"^[-a-zA-Z0-9_.]+$", dump_name):
                        msg = "Dump file name contains invalid characters"
                        raise ValueError(msg)

                    tmp_in_file.write(f"""
units metal
boundary p p p
atom_style atomic

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path_str}
pair_coeff * * zbl {zbl_elements}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt 10 v_max_gamma > {self.config.uncertainty_threshold} error hard

dump 1 all custom 10 {dump_name} id type x y z c_pace_gamma
run {self.config.md_steps}
""")
            # Atomically rename to target input file
            Path(tmp_path).replace(in_file)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        # Execute lammps
        try:
            # Lammps command line execution
            # 'lmp' or 'lmp_mpi' is standard
            # if we get an error, it might be the watchdog

            lmp_bin = shutil.which("lmp") or "lmp"
            subprocess.run(  # noqa: S603
                [lmp_bin, "-in", "in.lammps"],
                cwd=work_dir,
                check=True,
                capture_output=True,
                shell=False,
            )
        except subprocess.CalledProcessError:
            # If LAMMPS halted due to error hard
            logging.warning("LAMMPS halted, possibly due to uncertainty watchdog.")
            # For the pipeline logic, we assume it halted.
            if not dump_file.exists():
                msg = "LAMMPS halted but no dump file was generated."
                raise RuntimeError(msg) from None
            return {"halted": True, "dump_file": dump_file}
        except FileNotFoundError as e:
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from e
        else:
            return {"halted": False, "dump_file": None}

    def extract_high_gamma_structures(self, dump_file: Path, threshold: float) -> list[Atoms]:
        """Extracts structures with high gamma from LAMMPS dump."""
        # Read the trajectory.
        # This requires the dump to be in a readable format, e.g., custom format with gamma.
        if not dump_file.exists():
            msg = f"Dump file not found: {dump_file}"
            raise FileNotFoundError(msg)

        traj = read(str(dump_file), index=":", format="lammps-dump-text")  # type: ignore[no-untyped-call]
        if not isinstance(traj, list):
            traj = [traj]

        if not traj:
            msg = f"No structures read from dump file: {dump_file}"
            raise ValueError(msg)

        # Filter high gamma structures out of trajectory dump frames
        high_gamma = []
        for atoms in traj:
            if "c_pace_gamma" in atoms.arrays:
                # Find maximum gamma over atoms
                gamma_array = atoms.arrays["c_pace_gamma"]
                import numpy as np
                if np.max(gamma_array) > threshold:
                    high_gamma.append(atoms)
            else:
                # Default to last frame if gamma is not mapped, e.g. for mock testing or cold start
                pass

        if not high_gamma:
            return [traj[-1]]

        return high_gamma
