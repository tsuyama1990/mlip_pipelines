import re
from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
# Revert the horrible `Path("/home/jules").resolve(strict=True)_factory`
content = content.replace('Path("/home/jules").resolve(strict=True)_factory', 'tmp_path_factory')
content = content.replace('Path("/home/jules").resolve(strict=True) = Path("/home/jules/" + str(hash(Path.cwd())) + str(hash(__name__))).resolve(strict=True)', 'tmp_path = tmp_path_factory.mktemp("orchestrator")')

# Wait, `tmp_path` will be in `/tmp/...` which is NOT ALLOWED because of our new strict prefix checking?
# Actually, I removed `/tmp` from `restricted_prefixes` globally!
# So `/tmp` IS ALLOWED! `tmp_path` will work perfectly!
content = content.replace('Path("/home/jules/" + str(hash(Path.cwd())) + str(hash(__name__))).resolve(strict=True)', 'tmp_path_factory.mktemp("orchestrator")')

Path("tests/unit/test_orchestrator.py").write_text(content)
