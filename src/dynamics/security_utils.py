import os
import re
import shutil
import sys
from pathlib import Path


def _check_trusted_location(resolved_bin: Path, all_trusted: list[str]) -> None:
    is_trusted = False
    for td in all_trusted:
        try:
            td_resolved = Path(os.path.realpath(td)).resolve(strict=True)
            if resolved_bin.is_relative_to(td_resolved):
                is_trusted = True
                break
        except OSError:
            continue

    if not is_trusted:
        msg = f"Resolved binary must reside in a trusted directory: {resolved_bin}"
        raise ValueError(msg)


def _verify_executable_hash(resolved_bin: Path, expected_hash: str) -> None:
    import hashlib

    h = hashlib.sha256()
    with Path.open(resolved_bin, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    if h.hexdigest() != expected_hash:
        msg = f"Executable hash mismatch for {resolved_bin}"
        raise ValueError(msg)


def validate_executable_path(
    executable_name: str,
    trusted_directories: list[str],
    project_root: str | None = None,
    expected_hash: str | None = None,
) -> Path:
    """
    Validates that an executable path is safe to use.
    Returns the resolved absolute path as a Path object if safe, raises ValueError otherwise.
    """

    if not re.match(r"^[/a-zA-Z0-9_.-]+$", executable_name) or ".." in executable_name:
        msg = "Invalid characters in executable name"
        raise ValueError(msg)

    resolved_which: str | None = shutil.which(executable_name)
    if resolved_which is None:
        msg = f"Executable not found: {executable_name}"
        raise RuntimeError(msg)

    resolved_bin: Path = Path(os.path.realpath(resolved_which)).resolve(strict=True)

    # Do not allow resolving via symlinks that point outside trusted domains; explicitly fail if the base was a symlink.
    if Path(resolved_which).is_symlink():
        msg = "Binary cannot be a symlink."
        raise ValueError(msg)

    if not resolved_bin.is_file() or not os.access(resolved_bin, os.X_OK):
        msg = f"Binary is not an executable file: {resolved_bin}"
        raise ValueError(msg)

    all_trusted: list[str] = trusted_directories.copy()
    all_trusted.append(str(Path(sys.prefix) / "bin"))
    if project_root:
        all_trusted.append(str(Path(project_root) / "bin"))

    _check_trusted_location(resolved_bin, all_trusted)

    if expected_hash:
        _verify_executable_hash(resolved_bin, expected_hash)

    return resolved_bin.absolute()


def validate_filename(filename: str, extra_allowed_chars: str = "") -> None:
    """Validates that a filename is alphanumeric with standard safe characters."""
    pattern = f"^[a-zA-Z0-9_.-{extra_allowed_chars}]+$"
    if not re.match(pattern, filename):
        msg = f"Invalid characters in filename: {filename}"
        raise ValueError(msg)
