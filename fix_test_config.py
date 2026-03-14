import re

with open("tests/unit/test_config.py", "r") as f:
    content = f.read()

content = content.replace("from src.domain_models.config import _validate_single_trusted_dir", "from src.domain_models.config import _secure_resolve_and_validate_dir as _validate_single_trusted_dir")
content = content.replace("res = _validate_single_trusted_dir(str(d))", "res = _validate_single_trusted_dir(str(d), check_exists=True)")
content = content.replace("_validate_single_trusted_dir(str(d))", "_validate_single_trusted_dir(str(d), check_exists=True)")

with open("tests/unit/test_config.py", "w") as f:
    f.write(content)
