import re
from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
# I won't replace anything that spans multiple lines or creates them. I will just rely on `exist_ok=True` for `pot_dir` and delete the files inside it.
content = re.sub(
    r"pot_dir\.mkdir\(parents=True\)",
    'pot_dir.mkdir(parents=True, exist_ok=True); [f.unlink() for f in pot_dir.glob("*")]',
    content,
)
content = re.sub(
    r"tmp_work_dir\.mkdir\(parents=True\)",
    'tmp_work_dir.mkdir(parents=True, exist_ok=True); [f.unlink() for f in tmp_work_dir.glob("*")]',
    content,
)
content = re.sub(
    r"pot_dir\.mkdir\(\)",
    'pot_dir.mkdir(parents=True, exist_ok=True); [f.unlink() for f in pot_dir.glob("*")]',
    content,
)

# Fix name errors
content = content.replace("project_root = tmp_path", 'project_root = "/home/jules"')
Path("tests/unit/test_orchestrator.py").write_text(content)
