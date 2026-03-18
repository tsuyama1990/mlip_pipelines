import re
from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
# Let's fix the invalid syntax by properly formatting the tests/unit/test_orchestrator.py file.
# The issue is we injected `if pot_dir.exists():\n            import shutil\n            shutil.rmtree(pot_dir)\n` earlier and didn't remove it properly when we did the replacement in `patch_orch_final4.py`
content = re.sub(r'        if pot_dir.exists\(\):\n            import shutil\n            shutil\.rmtree\(pot_dir\)\n        import shutil; shutil\.rmtree\(pot_dir\) if pot_dir\.exists\(\) else None; import shutil; shutil\.rmtree\(pot_dir\) if pot_dir\.exists\(\) else None; pot_dir\.mkdir\(parents=True, exist_ok=True\); \[f\.unlink\(\) for f in pot_dir\.glob\("\*"\)\]', '        pot_dir.mkdir(parents=True, exist_ok=True)\n        [f.unlink() for f in pot_dir.glob("*")]', content)
content = re.sub(r'            if pot_dir.exists\(\):\n            import shutil\n            shutil\.rmtree\(pot_dir\)\n        import shutil; shutil\.rmtree\(pot_dir\) if pot_dir\.exists\(\) else None; import shutil; shutil\.rmtree\(pot_dir\) if pot_dir\.exists\(\) else None; pot_dir\.mkdir\(parents=True, exist_ok=True\); \[f\.unlink\(\) for f in pot_dir\.glob\("\*"\)\]', '        pot_dir.mkdir(parents=True, exist_ok=True)\n        [f.unlink() for f in pot_dir.glob("*")]', content)

content = re.sub(r'        if tmp_work_dir\.exists\(\):\n            import shutil\n            shutil\.rmtree\(tmp_work_dir\)\n        import shutil; shutil\.rmtree\(tmp_work_dir\) if tmp_work_dir\.exists\(\) else None; tmp_work_dir\.mkdir\(parents=True, exist_ok=True\); \[f\.unlink\(\) for f in tmp_work_dir\.glob\("\*"\)\]', '        tmp_work_dir.mkdir(parents=True, exist_ok=True)\n        [f.unlink() for f in tmp_work_dir.glob("*")]', content)
content = re.sub(r'            if tmp_work_dir\.exists\(\):\n            import shutil\n            shutil\.rmtree\(tmp_work_dir\)\n        import shutil; shutil\.rmtree\(tmp_work_dir\) if tmp_work_dir\.exists\(\) else None; tmp_work_dir\.mkdir\(parents=True, exist_ok=True\); \[f\.unlink\(\) for f in tmp_work_dir\.glob\("\*"\)\]', '        tmp_work_dir.mkdir(parents=True, exist_ok=True)\n        [f.unlink() for f in tmp_work_dir.glob("*")]', content)

Path("tests/unit/test_orchestrator.py").write_text(content)
