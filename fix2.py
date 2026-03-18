import re

with open("src/dynamics/dynamics_engine.py", "r") as f:
    text = f.read()

# Fix f-string templating in Dynamics Engine
# Replace with safe list construction and join to avoid injection entirely?
# Actually, the auditor specifically asks to "Replace f-string interpolation with a template-based approach using a strict whitelist of allowed values. Validate all configuration parameters against strict regex patterns before formatting the script."
# Wait, I had changed it from Template to f-string because the auditor complained about Template.
# Now they complain about f-string. "Dynamic script generation using f-strings with unvalidated configuration values creates a severe command injection vulnerability."
# The fix is: "Validate all configuration parameters against strict regex patterns before formatting the script."

# Let's add strict validation before the f-string for _write_cold_start_input, _write_potential_input, and _write_resume_input

replacement_cold = '''    def _write_cold_start_input(self, tmp_in_file: Any, dump_name: str, work_dir: Path) -> None:
        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(work_dir), check_exists=False)

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

        safe_lattice_size = float(self.config.lattice_size)
        safe_box_x = int(box_x)
        safe_box_y = int(box_y)
        safe_box_z = int(box_z)
        safe_md_steps = int(min(self.config.md_steps, 1000))

        script = f"""units metal
boundary p p p
atom_style atomic

lattice {lattice_type} {safe_lattice_size}
region box block 0 {safe_box_x} 0 {safe_box_y} 0 {safe_box_z}
create_box 2 box
create_atoms 1 box

# Cold start: using only ZBL
pair_style zbl 1.0 2.0
pair_coeff * * {zbl_mapping}

# Force dump to extract structures for initial training
dump 1 all custom 10 {dump_name} id type x y z
run {safe_md_steps}
write_restart {work_dir_str}/restart.lammps
write_data {work_dir_str}/data.lammps
"""
        tmp_in_file.write(script + "\\n")'''

text = re.sub(
    r'    def _write_cold_start_input\(self, tmp_in_file: Any, dump_name: str, work_dir: Path\) -> None:[\s\S]*?tmp_in_file\.write\(script \+ "\\n"\)',
    replacement_cold,
    text, count=1
)


replacement_pot = '''    def _write_potential_input(
        self, tmp_in_file: Any, potential: Path, dump_name: str, work_dir: Path
    ) -> None:
        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(work_dir), check_exists=False)

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

        safe_lattice_size = float(self.config.lattice_size)
        safe_box_x = int(box_x)
        safe_box_y = int(box_y)
        safe_box_z = int(box_z)
        safe_smooth_steps = int(self.config.thresholds.smooth_steps)
        safe_threshold = float(self.config.thresholds.threshold_call_dft)
        safe_dump_steps = max(10, self.config.md_steps // 100)
        safe_md_steps = int(self.config.md_steps)

        script = f"""units metal
boundary p p p
atom_style atomic

lattice {lattice_type} {safe_lattice_size}
region box block 0 {safe_box_x} 0 {safe_box_y} 0 {safe_box_z}
create_box 2 box
create_atoms 1 box

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path_str}
pair_coeff * * zbl {zbl_mapping}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt {safe_smooth_steps} v_max_gamma > {safe_threshold} error hard message "AL_HALT"

dump 1 all custom {safe_dump_steps} {dump_name} id type x y z c_pace_gamma
run {safe_md_steps}
write_restart {work_dir_str}/restart.lammps
write_data {work_dir_str}/data.lammps
"""
        tmp_in_file.write(script + "\\n")'''

text = re.sub(
    r'    def _write_potential_input\([\s\S]*?tmp_in_file\.write\(script \+ "\\n"\)',
    replacement_pot,
    text, count=1
)

replacement_res = '''    def _write_resume_input(
        self,
        in_file: Path,
        potential: Path,
        restart_file: Path,
        dump_file_name: str,
        work_dir: Path,
    ) -> None:
        from src.domain_models.config import _secure_resolve_and_validate_dir

        _secure_resolve_and_validate_dir(str(work_dir), check_exists=False)

        zbl_elements = " ".join(
            str(atomic_numbers.get(el, 1)) for el in self.system_config.elements
        )
        if not re.match(r"^[0-9 ]+$", zbl_elements):
            msg = "Invalid characters in zbl_elements"
            raise ValueError(msg)

        pot_path_str = self._validate_potential_path(potential)

        base_dump_name = Path(dump_file_name).name
        if not re.match(r"^[a-zA-Z0-9_.-]+$", base_dump_name) or base_dump_name != dump_file_name:
            msg = "Dump file name contains invalid characters"
            raise ValueError(msg)

        work_dir_str = str(work_dir.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", work_dir_str) or ".." in work_dir_str:
            msg = "Invalid characters in work_dir"
            raise ValueError(msg)

        restart_file_str = str(restart_file.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", restart_file_str) or ".." in restart_file_str:
            msg = "Invalid characters in restart_file"
            raise ValueError(msg)

        safe_smooth_steps = int(self.config.thresholds.smooth_steps)
        safe_threshold = float(self.config.thresholds.threshold_call_dft)
        safe_temperature = float(self.config.temperature)
        safe_md_steps = int(self.config.md_steps)

        script = f"""read_restart {restart_file_str}

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path_str}
pair_coeff * * zbl {zbl_elements}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt {safe_smooth_steps} v_max_gamma > {safe_threshold} error hard message "AL_HALT"

fix soft_start all langevin {safe_temperature} {safe_temperature} 0.1 48279
run 100
unfix soft_start

dump 1 all custom 10 {dump_file_name} id type x y z c_pace_gamma
run {safe_md_steps}
write_restart {work_dir_str}/restart.lammps
write_data {work_dir_str}/data.lammps
"""
        with Path.open(in_file, "w") as f:
            f.write(script + "\\n")'''

text = re.sub(
    r'    def _write_resume_input\([\s\S]*?f\.write\(script \+ "\\n"\)',
    replacement_res,
    text, count=1
)

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(text)
