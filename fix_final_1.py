with open("src/dynamics/dynamics_engine.py") as f:
    content = f.read()

# Fix _write_potential_input duplicate resolution and string format timing
old_pot_input = r"""    def _write_potential_input(
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

        validate_filename(dump_name, extra_allowed_chars=".")"""

new_pot_input = r"""    def _write_potential_input(
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

        validate_filename(dump_name, extra_allowed_chars=".")"""

content = content.replace(old_pot_input, new_pot_input)

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(content)
