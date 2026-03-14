with open("src/dynamics/eon_wrapper.py") as f:
    content = f.read()

old_pace = """    def _write_pace_driver(self, work_dir: Path, potential: Path | None) -> None:
        pots_dir = work_dir / "potentials"
        pots_dir.mkdir(parents=True, exist_ok=True)
        driver_path = pots_dir / "pace_driver.py"

        # the pace_driver should be executable by python

        # Security: strictly validate the potential string before formatting it into the python template

        resolved_pot_str = ""
        if potential:
            resolved_pot = Path(os.path.realpath(potential)).resolve(strict=True)
            if self.config.project_root is not None:
                root = Path(os.path.realpath(self.config.project_root)).resolve(strict=True)
                if not resolved_pot.is_relative_to(root):
                    msg = f"Potential path must be within the project root: {resolved_pot}"
                    raise ValueError(msg)
            # Ensure the potential string itself doesn't contain injected template syntax
            resolved_pot_str = str(resolved_pot)
            if "{" in resolved_pot_str or "}" in resolved_pot_str or '"' in resolved_pot_str or "'" in resolved_pot_str:
                msg = "Potential path contains invalid characters"
                raise ValueError(msg)


        pot_str = repr(resolved_pot_str) if potential else "None"

        # Ensure executable doesn't break python logic
        import re
        import sys

        executable = sys.executable
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", executable):
            msg = "Invalid python executable path"
            raise ValueError(msg)

        driver_content = self.config.eon_driver_template.format(
            executable=executable,
            threshold=float(self.config.uncertainty_threshold),
            pot_str=pot_str,
        )
        with Path.open(driver_path, "w") as f:
            f.write(driver_content)
        driver_path.chmod(0o755)"""

new_pace = """    def _write_pace_driver(self, work_dir: Path, potential: Path | None) -> None:
        resolved_work = work_dir.resolve(strict=False)
        if self.config.project_root is not None:
            root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_work.is_relative_to(root):
                msg = f"Work directory must be within the project root: {resolved_work}"
                raise ValueError(msg)

        pots_dir = resolved_work / "potentials"
        pots_dir.mkdir(parents=True, exist_ok=True)
        driver_path = pots_dir / "pace_driver.py"

        resolved_pot_str = "None"
        if potential:
            resolved_pot = potential.resolve(strict=True)
            if self.config.project_root is not None:
                if not resolved_pot.is_relative_to(root):
                    msg = f"Potential path must be within the project root: {resolved_pot}"
                    raise ValueError(msg)
            # Safely escape for python code using repr() which escapes quotes securely
            resolved_pot_str = repr(str(resolved_pot))

        import re
        import sys

        executable = sys.executable
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", executable):
            msg = "Invalid python executable path"
            raise ValueError(msg)

        # Template formatting is still used but the inputs are rigorously canonicalized and Python-escaped via repr
        driver_content = self.config.eon_driver_template.format(
            executable=executable,
            threshold=float(self.config.uncertainty_threshold),
            pot_str=resolved_pot_str,
        )
        with Path.open(driver_path, "w") as f:
            f.write(driver_content)
        driver_path.chmod(0o755)"""

content = content.replace(old_pace, new_pace)

with open("src/dynamics/eon_wrapper.py", "w") as f:
    f.write(content)
