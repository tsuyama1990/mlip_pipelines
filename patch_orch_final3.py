import re
from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
# Replace ONLY exactly what is needed for `test_orchestrator.py`
content = re.sub(
    r"pot_dir\.mkdir\(parents=True\)",
    "import shutil; shutil.rmtree(pot_dir) if pot_dir.exists() else None; pot_dir.mkdir(parents=True)",
    content,
)
content = re.sub(
    r"tmp_work_dir\.mkdir\(parents=True\)",
    "import shutil; shutil.rmtree(tmp_work_dir) if tmp_work_dir.exists() else None; tmp_work_dir.mkdir(parents=True)",
    content,
)

Path("tests/unit/test_orchestrator.py").write_text(content)
