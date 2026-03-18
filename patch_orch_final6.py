import re
from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()

# Target these exact lines causing IndentationError. They are stranded due to earlier replacements.
content = re.sub(r'        import shutil\n        if pot_dir\.exists\(\):\n            shutil\.rmtree\(pot_dir\)', '', content)
content = re.sub(r'        import shutil\n        if tmp_work_dir\.exists\(\):\n            shutil\.rmtree\(tmp_work_dir\)', '', content)

Path("tests/unit/test_orchestrator.py").write_text(content)
