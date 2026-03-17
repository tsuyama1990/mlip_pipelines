import hashlib
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

    if Path(executable_name).is_absolute():
        msg = "Executable name cannot be absolute path"
        raise ValueError(msg)

    if not re.match(VALID_EXECUTABLE_REGEX, executable_name) or ".." in executable_name:
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


# Security constants mapping to those needed by validators
VALID_ENV_KEY_REGEX = r"^[A-Z0-9_]+$"
VALID_EXECUTABLE_REGEX = r"^[/a-zA-Z0-9_.-]+$"


def _validate_env_key(key: str) -> None:
    if not key.startswith("MLIP_"):
        msg = f"Unauthorized environment variable injected via .env: {key}. Only MLIP_ prefixes are allowed."
        raise ValueError(msg)
    if len(key) > 64:
        msg = "Environment variable key exceeds maximum length"
        raise ValueError(msg)
    if not re.match(VALID_ENV_KEY_REGEX, key):
        msg = f"Invalid characters in .env variable key: {key}"
        raise ValueError(msg)


def _validate_env_value(val: str) -> None:
    if len(val) > 1024:
        msg = "Environment variable value exceeds maximum length"
        raise ValueError(msg)
    if ".." in val or ";" in val or "&" in val or "|" in val:
        msg = f"Invalid characters or traversal sequences in .env variable value: {val}."
        raise ValueError(msg)


def _validate_string_security(val: str) -> None:
    """Validates arbitrary strings to prevent path traversal and shell injection."""
    if not isinstance(val, str):
        msg = "Value must be a string."
        raise TypeError(msg)
    if len(val) > 256:
        msg = "String exceeds maximum allowed length (256 characters)."
        raise ValueError(msg)

    # Check for path traversals
    if ".." in val:
        msg = "Path traversal sequences (..) are not allowed."
        raise ValueError(msg)

    # Check for shell injection characters
    forbidden_chars = [";", "&", "|", "$", "`", "{", "}", "<", ">"]
    for char in forbidden_chars:
        if char in val:
            msg = f"Forbidden shell character '{char}' detected."
            raise ValueError(msg)


def validate_env_file_security(env_file: Path, expected_base: Path) -> Path:
    import stat

    if env_file.is_symlink():
        msg = ".env file must not be a symlink."
        raise ValueError(msg)

    resolved_env = env_file.resolve(strict=True)

    if resolved_env.parent != expected_base:
        msg = f".env file must reside directly in the allowed base directory: {expected_base}"
        raise ValueError(msg)

    st = os.lstat(resolved_env)

    if st.st_size > 10 * 1024:
        msg = ".env file exceeds maximum allowed size (10KB)."
        raise ValueError(msg)

    if st.st_uid != os.getuid():
        msg = ".env file is not owned by the current user."
        raise ValueError(msg)

    if bool(st.st_mode & stat.S_IRWXO) or bool(st.st_mode & stat.S_IRWXG):
        msg = ".env file has insecure permissions. It must not be group or world readable/writable."
        raise ValueError(msg)

    return resolved_env


def validate_and_copy_potential(
    src_pot: Path, pot_dir: Path, iteration: int, tmp_work_dir: Path
) -> Path:
    """Validates paths and securely copies a potential to the target directory."""
    import shutil

    final_dest = pot_dir / f"generation_{iteration:03d}.yace"

    src_resolved = src_pot.resolve(strict=True)
    if not src_resolved.is_relative_to(tmp_work_dir.resolve(strict=True)):
        msg = "Source potential must strictly reside within the tmp working directory"
        raise ValueError(msg)

    final_resolved = final_dest.resolve(strict=False)
    if not final_resolved.is_relative_to(pot_dir.resolve(strict=True)):
        msg = "Destination potential must strictly reside within the potentials directory"
        raise ValueError(msg)

    # Note: File size verification and hash verification should ideally happen here or before this step.
    # For now, we perform the atomic copy/move.
    shutil.copy2(src_resolved, final_resolved)
    return final_resolved
