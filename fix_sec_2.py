with open('src/dynamics/dynamics_engine.py', 'r') as f:
    content = f.read()

import re
# Oh, string.Template is used in _write_baseline_input and _write_potential_input! I only fixed one!

# Let's fix _write_baseline_input
content = re.sub(
    r'    def _write_baseline_input\(\n        self, tmp_in_file: Any, dump_name: str, work_dir: Path\n    \) -> None:\n        """Generates the LAMMPS input script fragment for exploring via the baseline potential\."""\n        import string.*?script = template\.substitute\(\n            lattice_type=lattice_type,\n            lattice_size=float\(self\.config\.lattice_size\),\n            box_x=int\(box_x\),\n            box_y=int\(box_y\),\n            box_z=int\(box_z\),\n            zbl_mapping=zbl_mapping,\n            dump_steps=max\(10, self\.config\.md_steps // 10\),\n            dump_name=dump_name,\n            md_steps=int\(self\.config\.md_steps\),\n            work_dir_str=work_dir_str,\n        \)\n        tmp_in_file\.write\(script \+ "\\n"\)',
    '''    def _write_baseline_input(
        self, tmp_in_file: Any, dump_name: str, work_dir: Path
    ) -> None:
        """Generates the LAMMPS input script fragment for exploring via the baseline potential."""
        base_dump_name = Path(dump_name).name
        if not re.match(r"^[a-zA-Z0-9_.-]+$", base_dump_name) or base_dump_name != dump_name:
            msg = "Dump file name contains invalid characters"
            raise ValueError(msg)

        box_x, box_y, box_z = self.config.box_size
        lattice_type = self.config.lattice_type

        # Extensive validation to prevent injection through configuration
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

        script = f"""units metal
boundary p p p
atom_style atomic

lattice {lattice_type} {float(self.config.lattice_size)}
region box block 0 {int(box_x)} 0 {int(box_y)} 0 {int(box_z)}
create_box 2 box
create_atoms 1 box

pair_style zbl 1.0 2.0
pair_coeff * * {zbl_mapping}

dump 1 all custom {max(10, self.config.md_steps // 10)} {dump_name} id type x y z
run {int(self.config.md_steps)}
write_restart {work_dir_str}/restart.lammps
write_data {work_dir_str}/data.lammps
"""
        tmp_in_file.write(script + "\\n")''',
    content,
    flags=re.DOTALL
)


# Let's fix _write_resume_input
content = re.sub(
    r'    def _write_resume_input\(\n        self, tmp_in_file: Any, potential: Path, restart_file: Path, dump_name: str, work_dir: Path\n    \) -> None:\n        """Generates LAMMPS input to resume from a restart file with a new potential\."""\n        import string.*?script = template\.substitute\(\n            restart_file_str=restart_file_str,\n            pot_path_str=pot_path_str,\n            zbl_mapping=zbl_mapping,\n            smooth_steps=self\.config\.thresholds\.smooth_steps,\n            threshold_call_dft=float\(self\.config\.thresholds\.threshold_call_dft\),\n            dump_steps=max\(10, self\.config\.md_steps // 100\),\n            dump_name=dump_name,\n            md_steps=int\(self\.config\.md_steps\),\n            work_dir_str=work_dir_str,\n        \)\n        tmp_in_file\.write\(script \+ "\\n"\)',
    '''    def _write_resume_input(
        self, tmp_in_file: Any, potential: Path, restart_file: Path, dump_name: str, work_dir: Path
    ) -> None:
        """Generates LAMMPS input to resume from a restart file with a new potential."""
        pot_path_str = self._validate_potential_path(potential)

        base_dump_name = Path(dump_name).name
        if not re.match(r"^[a-zA-Z0-9_.-]+$", base_dump_name) or base_dump_name != dump_name:
            msg = "Dump file name contains invalid characters"
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

        restart_file_str = str(restart_file.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", restart_file_str) or ".." in restart_file_str:
            msg = "Invalid characters in restart file path"
            raise ValueError(msg)

        zbl_mapping = self._get_zbl_mapping()
        if not re.match(r"^[0-9 ]+$", zbl_mapping):
            msg = "Invalid characters in zbl_mapping"
            raise ValueError(msg)

        script = f"""read_restart {restart_file_str}

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path_str}
pair_coeff * * zbl {zbl_mapping}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt {self.config.thresholds.smooth_steps} v_max_gamma > {float(self.config.thresholds.threshold_call_dft)} error hard message "AL_HALT"

dump 1 all custom {max(10, self.config.md_steps // 100)} {dump_name} id type x y z c_pace_gamma
run {int(self.config.md_steps)}
write_restart {work_dir_str}/restart.lammps
write_data {work_dir_str}/data.lammps
"""
        tmp_in_file.write(script + "\\n")''',
    content,
    flags=re.DOTALL
)

with open('src/dynamics/dynamics_engine.py', 'w') as f:
    f.write(content)
