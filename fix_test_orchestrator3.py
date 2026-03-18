from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
# Some places might have had fewer or more spaces. I will just use autopep8 or something to fix it.
import autopep8
content = autopep8.fix_code(content)
Path("tests/unit/test_orchestrator.py").write_text(content)
