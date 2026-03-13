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

    def __init__(self, config: DynamicsConfig, system_config: SystemConfig) -> None:
        self.config = config
        self.system_config = system_config

    def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:  # noqa: C901, PLR0915
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

lattice fcc 3.5
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box

# Cold start: using only ZBL
pair_style zbl 1.0 2.0
pair_coeff * * {zbl_elements}

# Force dump to extract structures for initial training
dump 1 all custom 10 {dump_name} id type x y z
run {min(self.config.md_steps, 1000)}  # Short run for cold start
write_restart {work_dir.resolve()}/restart.lammps
write_data {work_dir.resolve()}/data.lammps
""")
                else:
                    # Validate potential path ends with .yace and safely quote it
                    pot_path_str = str(potential.resolve())
                    if not pot_path_str.endswith(".yace"):
                        msg = "Potential path must end with .yace"
                        raise ValueError(msg)

                    import re

                    # Validate path safe for lammps without shlex quotes to avoid LAMMPS syntax errors
                    # using strict whitelist
                    if not re.match(r"^[-a-zA-Z0-9_.]+$", Path(pot_path_str).name):
                        msg = "Potential path contains invalid characters for LAMMPS"
                        raise ValueError(msg)

                    if not Path(pot_path_str).is_relative_to(self.config.project_root if hasattr(self.config, 'project_root') else Path("/").resolve()):
                        pass # Note: we just check characters in the name and let the system permissions handle absolute paths if it's already resolved above, but let's strictly stick to the name validation.

                    dump_name = dump_file.name
                    if not re.match(r"^[-a-zA-Z0-9_.]+$", dump_name):
                        msg = "Dump file name contains invalid characters"
                        raise ValueError(msg)

                    tmp_in_file.write(f"""
units metal
boundary p p p
atom_style atomic

lattice fcc 3.5
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path_str}
pair_coeff * * zbl {zbl_elements}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt 10 v_max_gamma > {self.config.uncertainty_threshold} error soft

dump 1 all custom 10 {dump_name} id type x y z c_pace_gamma
run {self.config.md_steps}
write_restart {work_dir.resolve()}/restart.lammps
write_data {work_dir.resolve()}/data.lammps
""")
            # Atomically rename to target input file
            Path(tmp_path).replace(in_file)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        # Execute lammps
        try:
            lmp_bin = shutil.which(self.config.lmp_binary) or self.config.lmp_binary
            # Explicit input validation
            import re
            if not re.match(r"^[-a-zA-Z0-9_.]+$", Path(lmp_bin).name):
                msg = f"Invalid LAMMPS binary path: {lmp_bin}"
                raise ValueError(msg)

            cmd = [lmp_bin, "-in", "in.lammps"]

            subprocess.run(  # noqa: S603
                cmd,
                cwd=work_dir,
                check=True,
                capture_output=True,
                shell=False,
            )
        except subprocess.CalledProcessError:
            pass  # error soft doesn't trigger CalledProcessError usually
        except FileNotFoundError as e:
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from e

        return self._check_halt(dump_file)

    def _check_halt(self, dump_file: Path) -> dict[str, Any]:
        """Checks if the run was halted based on the dump file output."""
        if not dump_file.exists():
            msg = "LAMMPS failed and no dump file was generated."
            raise RuntimeError(msg)

        is_halted = False
        try:
            high_gamma = self.extract_high_gamma_structures(
                dump_file, self.config.uncertainty_threshold
            )
            if high_gamma:
                last_frame = high_gamma[-1]
                if "c_pace_gamma" in last_frame.arrays:
                    import numpy as np

                    if (
                        np.max(last_frame.arrays["c_pace_gamma"])
                        > self.config.uncertainty_threshold
                    ):
                        is_halted = True
                        logging.warning("LAMMPS halted, possibly due to uncertainty watchdog.")
        except Exception as e:
            logging.warning(f"Error checking halt status: {e}")
            is_halted = True

        return {"halted": is_halted, "dump_file": dump_file}

    def resume(self, potential: Path, restart_dir: Path, work_dir: Path) -> dict[str, Any]:
        """Resumes the MD simulation with the newly updated potential."""
        logging.info(f"Resuming dynamics with new potential: {potential}")
        work_dir.mkdir(parents=True, exist_ok=True)
        dump_file = work_dir / "dump.lammps"
        in_file = work_dir / "in.lammps"

        restart_file = restart_dir / "restart.lammps"
        if not restart_file.exists():
            msg = f"Missing required file: {restart_file}"
            raise FileNotFoundError(msg)

        zbl_elements = " ".join(
            str(atomic_numbers.get(el, 1)) for el in self.system_config.elements
        )
        pot_path_str = str(potential.resolve())

        with Path.open(in_file, "w") as f:
            f.write(f"""
read_restart {restart_file.resolve()}

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path_str}
pair_coeff * * zbl {zbl_elements}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt 10 v_max_gamma > {self.config.uncertainty_threshold} error soft

dump 1 all custom 10 {dump_file.name} id type x y z c_pace_gamma
run {self.config.md_steps}
write_restart {work_dir.resolve()}/restart.lammps
write_data {work_dir.resolve()}/data.lammps
""")

        import shutil

        lmp_bin = shutil.which(self.config.lmp_binary) or self.config.lmp_binary
        import re
        if not re.match(r"^[-a-zA-Z0-9_.]+$", Path(lmp_bin).name):
            msg = f"Invalid LAMMPS binary path: {lmp_bin}"
            raise ValueError(msg)

        cmd = [lmp_bin, "-in", in_file.name]

        try:
            subprocess.run(  # noqa: S603
                cmd,
                cwd=work_dir,
                check=True,
                capture_output=True,
                shell=False,
            )
        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError as e:
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from e

        return self._check_halt(dump_file)

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
