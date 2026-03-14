with open("src/dynamics/dynamics_engine.py") as f:
    content = f.read()

old_exe = """    def _execute_lammps(self, work_dir: Path) -> None:  # noqa: C901, PLR0912
        # Secure input file definition by hardcoding it completely
        in_file_name = "in.lammps"
        validate_filename(in_file_name)
        resolved_in_file = Path(os.path.realpath(work_dir / in_file_name)).resolve(strict=False)
        resolved_work_dir = work_dir.resolve(strict=True)

        if not resolved_in_file.is_relative_to(resolved_work_dir):
            msg = f"Invalid input file name causing path traversal: {in_file_name}"
            raise ValueError(msg)"""

new_exe = """    def _execute_lammps(self, work_dir: Path) -> None:  # noqa: C901, PLR0912
        # Secure input file definition by hardcoding it completely
        in_file_name = "in.lammps"
        # Validate purely using canonical paths
        resolved_work_dir = work_dir.resolve(strict=True)
        resolved_in_file = (resolved_work_dir / in_file_name).resolve(strict=True)

        if not resolved_in_file.is_relative_to(resolved_work_dir):
            msg = "Input file must reside in work directory"
            raise ValueError(msg)"""

content = content.replace(old_exe, new_exe)

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(content)
