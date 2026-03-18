from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
content = content.replace('    if pot_dir.exists():\n', '')
content = content.replace('    if tmp_work_dir.exists():\n', '')
Path("tests/unit/test_orchestrator.py").write_text(content)
