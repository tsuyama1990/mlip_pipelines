import re

with open("src/domain_models/config.py", "r") as f:
    content = f.read()

content = re.sub(
    r'def _validate_env_value\(val: str\) -> None:\n    if len\(val\) > 1024:\n        msg = "Environment variable value exceeds maximum length"\n        raise TypeError\(msg\)\n    if "\.\." in val or ";" in val or "&" in val or "\|" in val:\n        msg = f"Invalid characters or traversal sequences in \.env variable value: \{val\}\."\n        raise TypeError\(msg\)',
    '''def _validate_env_value(val: str) -> None:
    if ".." in val:
        msg = f"Invalid traversal sequences in .env variable value: {val}."
        raise TypeError(msg)
    if not re.match(r"^[-a-zA-Z0-9_.:/=,+?&#@%]*$", val):
        msg = f"Invalid characters in .env variable value: {val}."
        raise TypeError(msg)''',
    content
)

content = re.sub(
    r'def _validate_env_key\(key: str\) -> None:\n    if not key\.startswith\("MLIP_"\):\n        msg = f"Unauthorized environment variable injected via \.env: \{key\}\. Only MLIP_ prefixes are allowed\."\n        raise TypeError\(msg\)\n    if len\(key\) > 64:\n        msg = "Environment variable key exceeds maximum length"\n        raise TypeError\(msg\)\n    if not re\.match\(VALID_ENV_KEY_REGEX, key\):\n        msg = f"Invalid characters in \.env variable key: \{key\}"\n        raise TypeError\(msg\)',
    '''def _validate_env_key(key: str) -> None:
    if not key.startswith("MLIP_"):
        msg = f"Unauthorized environment variable injected via .env: {key}. Only MLIP_ prefixes are allowed."
        raise TypeError(msg)
    if not re.match(VALID_ENV_KEY_REGEX, key):
        msg = f"Invalid characters in .env variable key: {key}"
        raise TypeError(msg)''',
    content
)

with open("src/domain_models/config.py", "w") as f:
    f.write(content)

with open("src/dynamics/security_utils.py", "r") as f:
    content = f.read()

content = re.sub(
    r'def _validate_env_value\(val: str\) -> None:\n    if len\(val\) > 1024:\n        msg = "Environment variable value exceeds maximum length"\n        raise TypeError\(msg\)\n    if "\.\." in val or ";" in val or "&" in val or "\|" in val:\n        msg = f"Invalid characters or traversal sequences in \.env variable value: \{val\}\."\n        raise TypeError\(msg\)',
    '''def _validate_env_value(val: str) -> None:
    if ".." in val:
        msg = f"Invalid traversal sequences in .env variable value: {val}."
        raise TypeError(msg)
    if not re.match(r"^[-a-zA-Z0-9_.:/=,+?&#@%]*$", val):
        msg = f"Invalid characters in .env variable value: {val}."
        raise TypeError(msg)''',
    content
)

content = re.sub(
    r'def _validate_env_key\(key: str\) -> None:\n    if not key\.startswith\("MLIP_"\):\n        msg = f"Unauthorized environment variable injected via \.env: \{key\}\. Only MLIP_ prefixes are allowed\."\n        raise TypeError\(msg\)\n    if len\(key\) > 64:\n        msg = "Environment variable key exceeds maximum length"\n        raise TypeError\(msg\)\n    if not re\.match\(VALID_ENV_KEY_REGEX, key\):\n        msg = f"Invalid characters in \.env variable key: \{key\}"\n        raise TypeError\(msg\)',
    '''def _validate_env_key(key: str) -> None:
    if not key.startswith("MLIP_"):
        msg = f"Unauthorized environment variable injected via .env: {key}. Only MLIP_ prefixes are allowed."
        raise TypeError(msg)
    if not re.match(VALID_ENV_KEY_REGEX, key):
        msg = f"Invalid characters in .env variable key: {key}"
        raise TypeError(msg)''',
    content
)

content = re.sub(
    r'VALID_ENV_KEY_REGEX = r"\^\[A-Z0-9_\]\+\$"',
    'VALID_ENV_KEY_REGEX = r"^[A-Z0-9_]+$"',
    content
)

content = re.sub(
    r'def validate_filename\(filename: str, extra_allowed_chars: str = ""\) -> None:\n    """Validates that a filename is alphanumeric with standard safe characters\."""\n    pattern = f"\^\[a-zA-Z0-9_\.-\{extra_allowed_chars\}\]\+\$"\n    if not re\.match\(pattern, filename\):\n        msg = f"Invalid characters in filename: \{filename\}"\n        raise ValueError\(msg\)',
    '''def validate_filename(filename: str) -> None:
    """Validates that a filename is alphanumeric with standard safe characters."""
    pattern = r"^[a-zA-Z0-9_.-]+$"
    if not re.match(pattern, filename):
        msg = f"Invalid characters in filename: {filename}"
        raise ValueError(msg)''',
    content
)

content = content.replace(
    'if Path(resolved_which).is_symlink():\n        msg = "Binary cannot be a symlink."\n        raise ValueError(msg)',
    '''if Path(executable_name).is_symlink() or Path(resolved_which).is_symlink():
        msg = "Binary cannot be a symlink."
        raise ValueError(msg)'''
)

with open("src/dynamics/security_utils.py", "w") as f:
    f.write(content)
