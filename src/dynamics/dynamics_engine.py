import logging
import os
import re
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
        base_dump_name = Path(dump_name).name
        if not re.match(r"^[a-zA-Z0-9_.-]+$", base_dump_name) or base_dump_name != dump_name:
            msg = "Dump file name contains invalid characters"
            raise ValueError(msg)

        box_x, box_y, box_z = self.config.box_size

        # Security: validate all template variables against strict whitelists before formatting
        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        if not isinstance(self.config.lattice_size, (float, int)) or self.config.lattice_size <= 0:
            msg = "Invalid lattice_size"
            raise ValueError(msg)

        if not all(isinstance(dim, int) and dim > 0 for dim in (box_x, box_y, box_z)):
            msg = "Invalid box sizes"
            raise ValueError(msg)

        if not isinstance(self.config.md_steps, int) or self.config.md_steps <= 0:
            msg = "Invalid md_steps"
            raise ValueError(msg)

        work_dir_str = str(work_dir.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", work_dir_str) or ".." in work_dir_str:
            msg = "Invalid characters in work_dir"
            raise ValueError(msg)

        zbl_mapping = self._get_zbl_mapping()
        if not re.match(r"^[0-9 ]+$", zbl_mapping):
            msg = "Invalid characters in zbl_mapping"
            raise ValueError(msg)

        script_lines = [
            "units metal",
            "boundary p p p",
            "atom_style atomic",
            "",
            f"lattice {lattice_type} {float(self.config.lattice_size)}",
            f"region box block 0 {int(box_x)} 0 {int(box_y)} 0 {int(box_z)}",
            "create_box 2 box",
            "create_atoms 1 box",
            "",
            "# Cold start: using only ZBL",
            "pair_style zbl 1.0 2.0",
            f"pair_coeff * * {zbl_mapping}",
            "",
            "# Force dump to extract structures for initial training",
            f"dump 1 all custom 10 {dump_name} id type x y z",
            f"run {int(min(self.config.md_steps, 1000))}",
            f"write_restart {work_dir_str}/restart.lammps",
            f"write_data {work_dir_str}/data.lammps",
        ]
        tmp_in_file.write("\n".join(script_lines) + "\n")

    def _validate_potential_path(self, potential: Path) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+\.yace$", potential.name):
            msg = "Potential path must be a valid .yace file"
            raise ValueError(msg)

        resolved_pot = potential.resolve(strict=True)

        if self.config.project_root:
            project_root_str = str(self.config.project_root)
            root = Path(os.path.realpath(project_root_str)).resolve(strict=True)
            if not resolved_pot.is_relative_to(root):
                msg = f"Potential path must be within the project root: {resolved_pot}"
                raise ValueError(msg)

        return str(resolved_pot)

    def _write_potential_input(
        self, tmp_in_file: Any, potential: Path, dump_name: str, work_dir: Path
    ) -> None:
        pot_path_str = self._validate_potential_path(potential)

        base_dump_name = Path(dump_name).name
        if not re.match(r"^[a-zA-Z0-9_.-]+$", base_dump_name) or base_dump_name != dump_name:
            msg = "Dump file name contains invalid characters"
            raise ValueError(msg)

        box_x, box_y, box_z = self.config.box_size

        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        if not isinstance(self.config.lattice_size, (float, int)) or self.config.lattice_size <= 0:
            msg = "Invalid lattice_size"
            raise ValueError(msg)

        if not all(isinstance(dim, int) and dim > 0 for dim in (box_x, box_y, box_z)):
            msg = "Invalid box sizes"
            raise ValueError(msg)

        if not isinstance(self.config.md_steps, int) or self.config.md_steps <= 0:
            msg = "Invalid md_steps"
            raise ValueError(msg)

        if not isinstance(self.config.uncertainty_threshold, (float, int)):
            msg = "Invalid uncertainty_threshold"
            raise TypeError(msg)

        work_dir_str = str(work_dir.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", work_dir_str) or ".." in work_dir_str:
            msg = "Invalid characters in work_dir"
            raise ValueError(msg)

        zbl_mapping = self._get_zbl_mapping()
        if not re.match(r"^[0-9 ]+$", zbl_mapping):
            msg = "Invalid characters in zbl_mapping"
            raise ValueError(msg)

        script_lines = [
            "units metal",
            "boundary p p p",
            "atom_style atomic",
            "",
            f"lattice {lattice_type} {float(self.config.lattice_size)}",
            f"region box block 0 {int(box_x)} 0 {int(box_y)} 0 {int(box_z)}",
            "create_box 2 box",
            "create_atoms 1 box",
            "",
            "pair_style hybrid/overlay pace zbl 1.0 2.0",
            f"pair_coeff * * pace {pot_path_str}",
            f"pair_coeff * * zbl {zbl_mapping}",
            "",
            "compute pace_gamma all pace gamma_mode=1",
            "variable max_gamma equal max(c_pace_gamma)",
            f"fix watchdog all halt 10 v_max_gamma > {float(self.config.uncertainty_threshold)} error soft",
            "",
            f"dump 1 all custom 10 {dump_name} id type x y z c_pace_gamma",
            f"run {int(self.config.md_steps)}",
            f"write_restart {work_dir_str}/restart.lammps",
            f"write_data {work_dir_str}/data.lammps",
        ]
        tmp_in_file.write("\n".join(script_lines) + "\n")

    def _execute_lammps(self, work_dir: Path) -> None:
        in_file_name = "in.lammps"
        validate_filename(in_file_name)
        resolved_in_file = Path(os.path.realpath(work_dir / in_file_name)).resolve(strict=True)
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
                expected_hash=self.config.binary_hashes.get(self.config.lmp_binary),
            )
        except RuntimeError as e:
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from e

        if not re.match(r"^[/a-zA-Z0-9_.-]+$", lmp_bin):
            msg = "Invalid characters in resolved LAMMPS binary path"
            raise ValueError(msg)

        cmd: list[str] = [lmp_bin, "-in", in_file_name]

        for arg in cmd:
            if not isinstance(arg, str):
                msg = f"Invalid command argument type: {type(arg)}"
                raise TypeError(msg)
            if not re.match(r"^[/a-zA-Z0-9_.-]+$", arg):
                msg = f"Invalid characters in command argument: {arg}"
                raise ValueError(msg)

        try:
            _res: subprocess.CompletedProcess[bytes] = subprocess.run(  # noqa: S603
                cmd,
                cwd=str(work_dir.resolve(strict=True)),
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
        if self.config.project_root:
            project_root_str = str(self.config.project_root)
            trusted_dirs.append(str(Path(project_root_str) / "bin"))

        validate_filename(in_file.name)

        project_root = self.config.project_root
        try:
            lmp_bin = validate_executable_path(
                self.config.lmp_binary,
                self.config.trusted_directories,
                project_root=str(project_root) if project_root else None,
                expected_hash=self.config.binary_hashes.get(self.config.lmp_binary),
            )
        except RuntimeError as e:
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from e

        if not re.match(r"^[/a-zA-Z0-9_.-]+$", lmp_bin):
            msg = "Invalid characters in resolved LAMMPS binary path"
            raise ValueError(msg)

        cmd = [lmp_bin, "-in", in_file.name]

        for arg in cmd:
            if not isinstance(arg, str):
                msg = f"Invalid command argument type: {type(arg)}"
                raise TypeError(msg)
            if not re.match(r"^[/a-zA-Z0-9_.-]+$", arg):
                msg = f"Invalid characters in command argument: {arg}"
                raise ValueError(msg)

        try:
            subprocess.run(  # noqa: S603
                cmd,
                cwd=str(work_dir.resolve(strict=True)),
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
