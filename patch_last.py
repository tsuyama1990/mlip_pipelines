import re
from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
# Replace the multiple shutil imports on one line that I introduced and fix indentation.
# Instead of dealing with the regex mess, I'll just write a clean function to sanitize it.
import ast

try:
    ast.parse(content)
except SyntaxError as e:
    print(f"Syntax error at {e.lineno}")

# Replace the mangled lines
content = re.sub(r'        import shutil; shutil\.rmtree\(pot_dir\) if pot_dir\.exists\(\) else None.*', '        import shutil\n        if pot_dir.exists():\n            shutil.rmtree(pot_dir)\n        pot_dir.mkdir(parents=True, exist_ok=True)', content)

content = re.sub(r'        if pot_dir\.exists\(\):\n            import shutil\n            shutil\.rmtree\(pot_dir\)', '', content)

Path("tests/unit/test_orchestrator.py").write_text(content)
import autopep8

Path("tests/unit/test_orchestrator.py").write_text(autopep8.fix_code(Path("tests/unit/test_orchestrator.py").read_text()))
