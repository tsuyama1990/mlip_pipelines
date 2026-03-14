with open("src/dynamics/security_utils.py") as f:
    content = f.read()

old_validate_executable = """    resolved_which: str | None = shutil.which(executable_name)
    if resolved_which is None:
        msg = f"Executable not found: {executable_name}"
        raise RuntimeError(msg)

    # Convert to Path and canonicalize
    resolved_bin: Path = Path(os.path.realpath(resolved_which)).resolve(strict=True)

    # 1. Reject symlinks that bypass validation
    if Path(resolved_which).is_symlink():
        msg = "Binary cannot be a symlink."
        raise ValueError(msg)"""

new_validate_executable = """    resolved_which: str | None = shutil.which(executable_name)
    if resolved_which is None:
        msg = f"Executable not found: {executable_name}"
        raise RuntimeError(msg)

    p_which = Path(resolved_which)

    # 1. Reject symlinks that bypass validation before canonicalizing
    if p_which.is_symlink():
        msg = "Binary cannot be a symlink."
        raise ValueError(msg)

    # Convert to Path and canonicalize securely
    resolved_bin: Path = p_which.resolve(strict=True)"""

content = content.replace(old_validate_executable, new_validate_executable)

with open("src/dynamics/security_utils.py", "w") as f:
    f.write(content)
