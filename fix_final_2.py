with open("src/dynamics/dynamics_engine.py") as f:
    content = f.read()

# Fix _write_cold_start_input work_dir validation
old_cold_input = """        lattice_type = self.config.lattice_type
        if not re.match(r"^[a-zA-Z0-9]+$", lattice_type):
            msg = "Invalid lattice_type"
            raise ValueError(msg)

        resolved_work = work_dir.resolve(strict=False)
        if self.config.project_root is not None:
            root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_work.is_relative_to(root):
                msg = f"Work directory must be within project root: {work_dir}"
                raise ValueError(msg)

        work_dir_str = str(work_dir.resolve(strict=True))
"""

new_cold_input = """        lattice_type = self.config.lattice_type
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
"""

content = content.replace(old_cold_input, new_cold_input)

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(content)
