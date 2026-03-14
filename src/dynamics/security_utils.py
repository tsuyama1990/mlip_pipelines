import os
import re
import shutil
import sys
from pathlib import Path


def validate_executable_path(  # noqa: C901

    executable_name: str,
    trusted_directories: list[str],
    project_root: str | None = None,
) -> str:
    """
    Validates that an executable path is safe to use.
    Returns the resolved absolute path as a string if safe, raises ValueError otherwise.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", executable_name):
        msg = f"Invalid binary name: {executable_name}"
        raise ValueError(msg)

    resolved_which = shutil.which(executable_name)
    if resolved_which is None:
        msg = f"Executable not found: {executable_name}"
        raise RuntimeError(msg)

    resolved_bin = Path(os.path.realpath(resolved_which)).resolve(strict=True)

    if Path(resolved_which).is_symlink():
        msg = "Binary cannot be a symlink."
        raise ValueError(msg)

    if not resolved_bin.is_file() or not os.access(resolved_bin, os.X_OK):
        msg = f"Binary is not an executable file: {resolved_bin}"
        raise ValueError(msg)

    valid_names = ["lmp", "lammps", "eonclient"]
    if resolved_bin.name not in valid_names:
        msg = f"Resolved binary name must be one of {valid_names}, got '{resolved_bin.name}'"
        raise ValueError(msg)

    all_trusted = trusted_directories.copy()
    all_trusted.append(str(Path(sys.prefix) / "bin"))
    if project_root:
        all_trusted.append(str(Path(project_root) / "bin"))

    is_trusted = False
    for td in all_trusted:
        try:
            if resolved_bin.is_relative_to(Path(td).resolve(strict=True)):
                is_trusted = True
                break
        except OSError:
            continue

    if not is_trusted:
        msg = f"Resolved binary must reside in a trusted directory: {resolved_bin}"
        raise ValueError(msg)

    return str(resolved_bin.absolute())
