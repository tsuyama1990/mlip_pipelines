from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
lines = content.split('\n')
for i, line in enumerate(lines):
    if line.strip() == "if pot_dir.exists():":
        lines[i] = "        if pot_dir.exists():"
    if line.strip() == "import shutil":
        lines[i] = "            import shutil"
    if line.strip() == "shutil.rmtree(pot_dir)":
        lines[i] = "            shutil.rmtree(pot_dir)"

content = "\n".join(lines)
Path("tests/unit/test_orchestrator.py").write_text(content)
