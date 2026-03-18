import re

# 1. config.py regex fix
with open("src/domain_models/config.py", "r") as f:
    text = f.read()

# Replace _validate_env_value
replacement = '''def _validate_env_value(val: str) -> None:
    if not re.match(r"^[-a-zA-Z0-9_.:/=]{1,1024}$", val):
        msg = "Invalid characters detected in .env variable value."
        raise ValueError(msg)'''

text = re.sub(
    r'def _validate_env_value\(val: str\) -> None:\n    if not re.match\(r"\^\[-a-zA-Z0-9_\.:/=\]\{1,1024\}\$", val\):\n        msg = "Invalid characters detected in \.env variable value\."\n        raise ValueError\(msg\)',
    replacement,
    text
)
# And the complex block in _validate_env_file_security
replacement2 = '''                if not re.match(r"^[-a-zA-Z0-9_.:/=]{1,1024}$", val):
                    msg = f"Invalid characters or traversal sequences in .env file content: {val}"
                    raise ValueError(msg)'''

text = re.sub(
    r'                if \(\n                    "\.\." in val\n.*?\n                \):\n                    msg = f"Invalid characters or traversal sequences in \.env file content: \{val\}"\n                    raise ValueError\(msg\)',
    replacement2,
    text, flags=re.DOTALL
)

with open("src/domain_models/config.py", "w") as f:
    f.write(text)
