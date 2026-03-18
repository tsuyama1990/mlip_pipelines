import autopep8
from pathlib import Path

content = Path("tests/uat/verify_cycle_02_domain_logic.py").read_text()
new_content = autopep8.fix_code(content)
Path("tests/uat/verify_cycle_02_domain_logic.py").write_text(new_content)
