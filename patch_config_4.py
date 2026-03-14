with open("src/domain_models/config.py") as f:
    content = f.read()

# Inline the .env validation to avoid circular import with security_utils
env_val_code = """    @model_validator(mode="before")
    @classmethod
    def validate_env_content(cls, values: dict[str, Any]) -> dict[str, Any]:
        expected_base = Path.cwd().resolve(strict=True)
        env_file = expected_base / ".env"

        if env_file.exists():
            if not env_file.is_file():
                msg = ".env must be a regular file."
                raise ValueError(msg)

            if env_file.is_symlink():
                msg = ".env file must not be a symlink."
                raise ValueError(msg)

            resolved_env = env_file.resolve(strict=True)

            # Additional content safety checks
            with Path.open(resolved_env, encoding="utf-8") as f:
                content = f.read()
                if "import " in content or "eval(" in content or "exec(" in content:
                    msg = "Suspicious content detected in .env file."
                    raise ValueError(msg)

            with Path.open(resolved_env, encoding="utf-8") as f:
                for raw_line in f:
                    clean_line = raw_line.strip()
                    if not clean_line or clean_line.startswith("#") or "=" not in clean_line:
                        continue

                    key, val = clean_line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')

                    cls._validate_env_key(key)
                    cls._validate_env_value(val)

        return values"""

import re

content = re.sub(r'    @model_validator\(mode="before"\)\n    @classmethod\n    def validate_env_content.*?return values', env_val_code, content, flags=re.DOTALL)

with open("src/domain_models/config.py", "w") as f:
    f.write(content)
