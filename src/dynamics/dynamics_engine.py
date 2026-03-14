import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read

from src.core import AbstractDynamics
from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.security_utils import validate_executable_path, validate_filename


class MDInterface(AbstractDynamics):
    """Manages LAMMPS execution using the python module or subprocess."""

    def __init__(self, config: DynamicsConfig, system_config: SystemConfig) -> None:
        self.config = config
        self.system_config = system_config

    def _get_zbl_mapping(self) -> str:
        return " ".join(str(atomic_numbers.get(el, 1)) for el in self.system_config.elements)

    def _write_cold_start_input(self, tmp_in_file: Any, dump_name: str, work_dir: Path) -> None:
        if not re.match(r"^[a-zA-Z0-9_]+(\.lammps)?$", dump_name):
            msg = "Dump file name contains invalid characters"
            raise ValueError(msg)

        box_x, box_y, box_z = self.config.box_size

        # Security: validate all template variables against strict whitelists before formatting
        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        work_dir_str = str(work_dir.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", work_dir_str) or ".." in work_dir_str:
            msg = "Invalid characters in work_dir"
            raise ValueError(msg)

        template = self.config.lammps_cold_start_template
        script = template.format(
            lattice_type=lattice_type,
            lattice_size=float(self.config.lattice_size),
            box_x=int(box_x),
            box_y=int(box_y),
            box_z=int(box_z),
            zbl_mapping=self._get_zbl_mapping(),
            dump_name=shlex.quote(dump_name),
            md_steps=int(min(self.config.md_steps, 1000)),
            work_dir=shlex.quote(work_dir_str),
        )
        tmp_in_file.write(script)

    def _write_potential_input(
        self, tmp_in_file: Any, potential: Path, dump_name: str, work_dir: Path
    ) -> None:
        resolved_pot = potential.resolve(strict=True)
        pot_path_str = str(resolved_pot)

        if not re.match(r"^[a-zA-Z0-9_]+\.yace$", potential.name):
            msg = "Potential path must be a valid .yace file"
            raise ValueError(msg)

        resolved_pot = potential.resolve(strict=True)
        pot_path_str = str(resolved_pot)

        # Verify the potential path is within the project root to prevent path traversal
        if self.config.project_root is not None:
            project_root_str = str(self.config.project_root)
            root = Path(os.path.realpath(project_root_str)).resolve(strict=True)
            if not resolved_pot.is_relative_to(root):
                msg = f"Potential path must be within the project root: {resolved_pot}"
                raise ValueError(msg)

        if not re.match(r"^[a-zA-Z0-9_]+(\.lammps)?$", dump_name):
            msg = "Dump file name contains invalid characters"
            raise ValueError(msg)

        template = self.config.lammps_script_template

        box_x, box_y, box_z = self.config.box_size

        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        work_dir_str = str(work_dir.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", work_dir_str) or ".." in work_dir_str:
            msg = "Invalid characters in work_dir"
            raise ValueError(msg)

        script = template.format(
            lattice_type=lattice_type,
            lattice_size=float(self.config.lattice_size),
            box_x=int(box_x),
            box_y=int(box_y),
            box_z=int(box_z),
            pot_path=shlex.quote(pot_path_str),
            zbl_mapping=self._get_zbl_mapping(),
            threshold=float(self.config.uncertainty_threshold),
            dump_name=shlex.quote(dump_name),
            md_steps=int(self.config.md_steps),
            work_dir=shlex.quote(work_dir_str),
        )
        tmp_in_file.write(script)

    def _execute_lammps(self, work_dir: Path) -> None:
        in_file_name = "in.lammps"
        validate_filename(in_file_name)
        resolved_in_file = Path(os.path.realpath(work_dir / in_file_name)).resolve(strict=False)
        resolved_work_dir = work_dir.resolve(strict=True)

        if not resolved_in_file.is_relative_to(resolved_work_dir):
            msg = f"Invalid input file name causing path traversal: {in_file_name}"
            raise ValueError(msg)


        project_root = self.config.project_root
        try:
            lmp_bin = validate_executable_path(
                self.config.lmp_binary,
                self.config.trusted_directories,
                project_root=str(project_root) if project_root else None,
            )
        except RuntimeError as e:
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from e

        cmd: list[str] = [lmp_bin, "-in", in_file_name]

        try:
            _res: subprocess.CompletedProcess[bytes] = subprocess.run(
                cmd,
                cwd=work_dir,
                check=True,
                capture_output=True,
                shell=False,
            )
        except subprocess.CalledProcessError:
            logging.info("LAMMPS run completed with a CalledProcessError (possibly soft halt).")
        except FileNotFoundError as e:
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from e

    def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:
        """Runs LAMMPS exploration and monitors for high uncertainty."""
        work_dir.mkdir(parents=True, exist_ok=True)
        dump_file = work_dir / "dump.lammps"

        if potential is not None and not potential.exists():
            msg = f"Potential file not found: {potential}"
            raise FileNotFoundError(msg)

        in_file = work_dir / "in.lammps"

        fd, tmp_path = tempfile.mkstemp(dir=work_dir, text=True)
        try:
            with os.fdopen(fd, "w") as tmp_in_file:
                if potential is None:
                    self._write_cold_start_input(tmp_in_file, dump_file.name, work_dir)
                else:
                    self._write_potential_input(tmp_in_file, potential, dump_file.name, work_dir)
            Path(tmp_path).replace(in_file)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        self._execute_lammps(work_dir)
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


        trusted_dirs = self.config.trusted_directories.copy()
        trusted_dirs.append(str(Path(sys.prefix) / "bin"))
        if self.config.project_root is not None:
            project_root_str = str(self.config.project_root)
            trusted_dirs.append(str(Path(project_root_str) / "bin"))

        validate_filename(in_file.name)


        project_root = self.config.project_root
        try:
            lmp_bin = validate_executable_path(
                self.config.lmp_binary,
                self.config.trusted_directories,
                project_root=str(project_root) if project_root else None,
            )
        except RuntimeError as e:
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from e

        cmd = [lmp_bin, "-in", in_file.name]

        try:
            subprocess.run(
                cmd,
                cwd=work_dir,
                check=True,
                capture_output=True,
                shell=False,
            )
        except subprocess.CalledProcessError:
            logging.info("LAMMPS resume run completed with a CalledProcessError.")
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
                # Default to last frame if gamma is not mapped, e.g. for cold start
                continue

        if not high_gamma:
            return [traj[-1]]

        return high_gamma
