with open("src/dynamics/dynamics_engine.py") as f:
    content = f.read()

# Replace _write_potential_input
old_pot = r"""    def _write_potential_input(
        self, tmp_in_file: Any, potential: Path, dump_name: str, work_dir: Path
    ) -> None:
        if not re.match(r"^[a-zA-Z0-9_]+\.yace$", potential.name):
            msg = "Potential path must be a valid .yace file"
            raise ValueError(msg)

        resolved_pot = potential.resolve(strict=True)

        # Verify the potential path is within the project root to prevent path traversal
        if self.config.project_root is not None:
            project_root_str = str(self.config.project_root)
            root = Path(os.path.realpath(project_root_str)).resolve(strict=True)
            if not resolved_pot.is_relative_to(root):
                msg = f"Potential path must be within the project root: {resolved_pot}"
                raise ValueError(msg)

        pot_path_str = str(resolved_pot)
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", pot_path_str):
            msg = "Potential path contains invalid characters"
            raise ValueError(msg)

        validate_filename(dump_name, extra_allowed_chars=".")

        template = self.config.lammps_script_template

        box_x, box_y, box_z = self.config.box_size

        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        work_dir_str = str(work_dir.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$" , work_dir_str) or ".." in work_dir_str:
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
        tmp_in_file.write(script)"""

new_pot = """    def _write_potential_input(
        self, tmp_in_file: Any, potential: Path, dump_name: str, work_dir: Path
    ) -> None:
        resolved_pot = potential.resolve(strict=True)

        if self.config.project_root is not None:
            root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_pot.is_relative_to(root):
                msg = f"Potential path must be within the project root: {resolved_pot}"
                raise ValueError(msg)

        resolved_work = work_dir.resolve(strict=True)
        if self.config.project_root is not None:
            root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_work.is_relative_to(root):
                msg = f"Work directory must be within project root: {resolved_work}"
                raise ValueError(msg)

        validate_filename(dump_name, extra_allowed_chars=".")

        template = self.config.lammps_script_template

        box_x, box_y, box_z = self.config.box_size

        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        script = template.format(
            lattice_type=lattice_type,
            lattice_size=float(self.config.lattice_size),
            box_x=int(box_x),
            box_y=int(box_y),
            box_z=int(box_z),
            pot_path=shlex.quote(str(resolved_pot)),
            zbl_mapping=self._get_zbl_mapping(),
            threshold=float(self.config.uncertainty_threshold),
            dump_name=shlex.quote(dump_name),
            md_steps=int(self.config.md_steps),
            work_dir=shlex.quote(str(resolved_work)),
        )
        tmp_in_file.write(script)"""

content = content.replace(old_pot, new_pot)

old_cold = """    def _write_cold_start_input(self, tmp_in_file: Any, dump_name: str, work_dir: Path) -> None:
        validate_filename(dump_name, extra_allowed_chars=".")

        box_x, box_y, box_z = self.config.box_size

        # Security: validate all template variables against strict whitelists before formatting
        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        resolved_work = work_dir.resolve(strict=True)
        if self.config.project_root is not None:
            root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_work.is_relative_to(root):
                msg = f"Work directory must be within project root: {resolved_work}"
                raise ValueError(msg)

        work_dir_str = str(resolved_work)

        template = self.config.lammps_script_template
        script = template.format(
            lattice_type=lattice_type,
            lattice_size=float(self.config.lattice_size),
            box_x=int(box_x),
            box_y=int(box_y),
            box_z=int(box_z),
            pot_path="none",
            zbl_mapping=self._get_zbl_mapping(),
            threshold=float(self.config.uncertainty_threshold),
            dump_name=shlex.quote(dump_name),
            md_steps=int(self.config.md_steps),
            work_dir=shlex.quote(work_dir_str),
        )
        tmp_in_file.write(script)"""

new_cold = """    def _write_cold_start_input(self, tmp_in_file: Any, dump_name: str, work_dir: Path) -> None:
        validate_filename(dump_name, extra_allowed_chars=".")

        box_x, box_y, box_z = self.config.box_size

        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        resolved_work = work_dir.resolve(strict=True)
        if self.config.project_root is not None:
            root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_work.is_relative_to(root):
                msg = f"Work directory must be within project root: {resolved_work}"
                raise ValueError(msg)

        template = self.config.lammps_script_template
        script = template.format(
            lattice_type=lattice_type,
            lattice_size=float(self.config.lattice_size),
            box_x=int(box_x),
            box_y=int(box_y),
            box_z=int(box_z),
            pot_path="none",
            zbl_mapping=self._get_zbl_mapping(),
            threshold=float(self.config.uncertainty_threshold),
            dump_name=shlex.quote(dump_name),
            md_steps=int(self.config.md_steps),
            work_dir=shlex.quote(str(resolved_work)),
        )
        tmp_in_file.write(script)"""

content = content.replace(old_cold, new_cold)

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(content)
