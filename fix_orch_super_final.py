import re
from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()

# We replace all variations of that completely messed up string.
bad1 = r'        import shutil; shutil\.rmtree\(pot_dir\) if pot_dir\.exists\(\) else None; import shutil; shutil\.rmtree\(pot_dir\) if pot_dir\.exists\(\) else None; pot_dir\.mkdir\(parents=True, exist_ok=True\); \[f\.unlink\(\) for f in pot_dir\.glob\("\*"\)\]'
bad2 = r'        import shutil; shutil\.rmtree\(tmp_work_dir\) if tmp_work_dir\.exists\(\) else None; import shutil; shutil\.rmtree\(tmp_work_dir\) if tmp_work_dir\.exists\(\) else None; import shutil; shutil\.rmtree\(tmp_work_dir\) if tmp_work_dir\.exists\(\) else None; import shutil; shutil\.rmtree\(tmp_work_dir\) if tmp_work_dir\.exists\(\) else None; tmp_work_dir\.mkdir\(parents=True, exist_ok=True\); \[f\.unlink\(\) for f in tmp_work_dir\.glob\("\*"\)\]'
bad3 = r'        if pot_dir\.exists\(\):\n            import shutil\n            shutil\.rmtree\(pot_dir\)'
bad4 = r'        if tmp_work_dir\.exists\(\):\n            import shutil\n            shutil\.rmtree\(tmp_work_dir\)'

content = re.sub(bad1, '        pot_dir.mkdir(parents=True, exist_ok=True)\n        [f.unlink() for f in pot_dir.glob("*")]', content)
content = re.sub(bad2, '        tmp_work_dir.mkdir(parents=True, exist_ok=True)\n        [f.unlink() for f in tmp_work_dir.glob("*")]', content)

# Remove the dangling ifs that were causing IndentationError because we replaced the line AFTER them but left them.
# The line AFTER them was empty or had the bad string.
content = re.sub(bad3, '', content)
content = re.sub(bad4, '', content)

# Remove dangling `import shutil`
content = re.sub(r'        import shutil\n\n', '', content)

Path("tests/unit/test_orchestrator.py").write_text(content)
