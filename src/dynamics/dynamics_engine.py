import logging
import os
import re
import subprocess
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
        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(work_dir), check_exists=False)
        import string

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

        template = string.Template("""units metal
boundary p p p
atom_style atomic

lattice ${lattice_type} ${lattice_size}
region box block 0 ${box_x} 0 ${box_y} 0 ${box_z}
create_box 2 box
create_atoms 1 box

# Cold start: using only ZBL
pair_style zbl 1.0 2.0
pair_coeff * * ${zbl_mapping}

# Force dump to extract structures for initial training
dump 1 all custom 10 ${dump_name} id type x y z
run ${md_steps_run}
write_restart ${work_dir_str}/restart.lammps
write_data ${work_dir_str}/data.lammps
""")

        script = template.substitute(
            lattice_type=lattice_type,
            lattice_size=float(self.config.lattice_size),
            box_x=int(box_x),
            box_y=int(box_y),
            box_z=int(box_z),
            zbl_mapping=zbl_mapping,
            dump_name=dump_name,
            md_steps_run=int(min(self.config.md_steps, 1000)),
            work_dir_str=work_dir_str,
        )

        tmp_in_file.write(script + "\n")

    def _validate_potential_path(self, potential: Path) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+\.yace$", potential.name):
            msg = "Potential path must be a valid .yace file"
            raise ValueError(msg)

        import os

        pot_real = os.path.realpath(str(potential))
        if ".." in pot_real:
            msg = f"Path traversal characters detected in potential path: {potential}"
            raise ValueError(msg)

        resolved_pot = Path(pot_real).resolve(strict=True)

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
        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(work_dir), check_exists=False)
        import string

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

        template = string.Template("""units metal
boundary p p p
atom_style atomic

lattice ${lattice_type} ${lattice_size}
region box block 0 ${box_x} 0 ${box_y} 0 ${box_z}
create_box 2 box
create_atoms 1 box

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace ${pot_path_str}
pair_coeff * * zbl ${zbl_mapping}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt ${smooth_steps} v_max_gamma > ${threshold_call_dft} error hard message "AL_HALT"

dump 1 all custom ${dump_steps} ${dump_name} id type x y z c_pace_gamma
run ${md_steps}
write_restart ${work_dir_str}/restart.lammps
write_data ${work_dir_str}/data.lammps
""")

        script = template.substitute(
            lattice_type=lattice_type,
            lattice_size=float(self.config.lattice_size),
            box_x=int(box_x),
            box_y=int(box_y),
            box_z=int(box_z),
            pot_path_str=pot_path_str,
            zbl_mapping=zbl_mapping,
            smooth_steps=self.config.thresholds.smooth_steps,
            threshold_call_dft=float(self.config.thresholds.threshold_call_dft),
            dump_steps=max(10, self.config.md_steps // 100),
            dump_name=dump_name,
            md_steps=int(self.config.md_steps),
            work_dir_str=work_dir_str,
        )

        tmp_in_file.write(script + "\n")

    def _parse_halt_log(self, log_file: Path) -> bool:
        """Parses LAMMPS log to check if halt was due to AL watchdog or fatal crash."""
        if not log_file.exists():
            return False

        try:
            with Path.open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096), 0)
                tail = f.read()
                return "AL_HALT" in tail
        except Exception:
            return False

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

        if not lmp_bin.is_absolute() or not lmp_bin.is_file():
            msg = "Invalid resolved LAMMPS binary path"
            raise ValueError(msg)

        cmd: list[str] = [str(lmp_bin), "-in", in_file_name]

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
        except subprocess.CalledProcessError as e:
            # Check if it was an intentional AL_HALT
            log_path = work_dir / "log.lammps"
            if self._parse_halt_log(log_path):
                logging.info("LAMMPS run correctly paused due to AL_HALT watchdog.")
            else:
                from src.core.exceptions import DynamicsHaltInterrupt

                msg = f"Fatal LAMMPS crash. log.lammps tail missing AL_HALT string. Code: {e.returncode}"
                raise DynamicsHaltInterrupt(msg) from e
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

    def _write_resume_input(
        self,
        in_file: Path,
        potential: Path,
        restart_file: Path,
        dump_file_name: str,
        work_dir: Path,
    ) -> None:
        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(work_dir), check_exists=False)
        import string

        zbl_elements = " ".join(
            str(atomic_numbers.get(el, 1)) for el in self.system_config.elements
        )
        pot_path_str = str(potential.resolve())

        template = string.Template("""read_restart ${restart_file}

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace ${pot_path_str}
pair_coeff * * zbl ${zbl_elements}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt ${smooth_steps} v_max_gamma > ${threshold_call_dft} error hard message "AL_HALT"

fix soft_start all langevin ${temperature} ${temperature} 0.1 48279
run 100
unfix soft_start

dump 1 all custom 10 ${dump_file_name} id type x y z c_pace_gamma
run ${md_steps}
write_restart ${work_dir_str}/restart.lammps
write_data ${work_dir_str}/data.lammps
""")

        script = template.substitute(
            restart_file=str(restart_file.resolve()),
            pot_path_str=pot_path_str,
            zbl_elements=zbl_elements,
            smooth_steps=self.config.thresholds.smooth_steps,
            threshold_call_dft=float(self.config.thresholds.threshold_call_dft),
            temperature=float(self.config.temperature),
            dump_file_name=dump_file_name,
            md_steps=int(self.config.md_steps),
            work_dir_str=str(work_dir.resolve()),
        )

        with Path.open(in_file, "w") as f:
            f.write(script + "\n")

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

        self._write_resume_input(in_file, potential, restart_file, dump_file.name, work_dir)
        self._execute_lammps(work_dir)

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
