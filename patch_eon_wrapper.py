import re

with open("src/dynamics/eon_wrapper.py") as f:
    data = f.read()

validate_potential_replacement = """    def _validate_potential_path(self, potential: Path) -> str:
        validate_filename(potential.name)

        if ".." in str(potential):
            msg = "Potential path contains invalid traversal characters."
            raise ValueError(msg)

        import os
        import stat

        try:
            st = os.lstat(potential)
        except Exception as e:
            msg = f"Failed to access potential file: {e}"
            raise ValueError(msg) from e

        if stat.S_ISLNK(st.st_mode):
            msg = "Potential file must not be a symlink."
            raise ValueError(msg)

        try:
            resolved_pot = Path(os.path.realpath(potential))
        except Exception as e:
            msg = f"Failed to resolve potential path: {e}"
            raise ValueError(msg) from e

        if not resolved_pot.is_file():
            msg = "Potential must be a regular file."
            raise ValueError(msg)

        if self.config.project_root:
            root = Path(os.path.realpath(self.config.project_root))
            if not resolved_pot.is_relative_to(root):
                msg = f"Potential path must be within the project root: {resolved_pot}"
                raise ValueError(msg)
        return resolved_pot.as_posix()"""

data = re.sub(
    r"    def _validate_potential_path.*?return resolved_pot.as_posix\(\)",
    validate_potential_replacement,
    data,
    flags=re.DOTALL,
)

with open("src/dynamics/eon_wrapper.py", "w") as f:
    f.write(data)
