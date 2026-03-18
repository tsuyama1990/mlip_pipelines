import re

with open("src/dynamics/dynamics_engine.py", "r") as f:
    text = f.read()

# Fix potential validation to be compliant with the tests
text = re.sub(
    r"    def _validate_potential_path\(self, potential: Path\) -> str:.*?        return str\(resolved_pot\)",
    r"""    def _validate_potential_path(self, potential: Path) -> str:
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

        return str(resolved_pot)""",
    text,
    flags=re.DOTALL,
)

# And make sure it is actually called by replacing back the deleted line
text = text.replace(
    "        base_dump_name = Path(dump_name).name",
    "        pot_path_str = self._validate_potential_path(potential)\n\n        base_dump_name = Path(dump_name).name",
)

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(text)
