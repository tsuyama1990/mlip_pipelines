from pathlib import Path
import re

content = Path("tests/unit/test_orchestrator.py").read_text()
# Let's fix FileExistsError
content = re.sub(r'pot_dir\.mkdir\(parents=True, exist_ok=True\)', 'try:\n        pot_dir.mkdir(parents=True, exist_ok=True)\n    except FileExistsError:\n        pass', content)
content = re.sub(r'tmp_work_dir\.mkdir\(parents=True, exist_ok=True\)', 'try:\n        tmp_work_dir.mkdir(parents=True, exist_ok=True)\n    except FileExistsError:\n        pass', content)
Path("tests/unit/test_orchestrator.py").write_text(content)
