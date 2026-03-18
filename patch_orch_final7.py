import re
from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
# Some statements have 8 spaces instead of 4
content = re.sub(r'^\s+pot_dir\.mkdir\(parents=True, exist_ok=True\)', '    pot_dir.mkdir(parents=True, exist_ok=True)', content, flags=re.MULTILINE)
content = re.sub(r'^\s+tmp_work_dir\.mkdir\(parents=True, exist_ok=True\)', '    tmp_work_dir.mkdir(parents=True, exist_ok=True)', content, flags=re.MULTILINE)
content = re.sub(r'^\s+\[f\.unlink\(\) for f in pot_dir\.glob\("\*"\)\]', '    [f.unlink() for f in pot_dir.glob("*")]', content, flags=re.MULTILINE)
content = re.sub(r'^\s+\[f\.unlink\(\) for f in tmp_work_dir\.glob\("\*"\)\]', '    [f.unlink() for f in tmp_work_dir.glob("*")]', content, flags=re.MULTILINE)

content = re.sub(r'            import shutil\n            shutil\.rmtree\(tmp_work_dir\)', '', content)

Path("tests/unit/test_orchestrator.py").write_text(content)
