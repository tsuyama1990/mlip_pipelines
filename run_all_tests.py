import subprocess
from pathlib import Path

p = Path("dev_documents")
p.mkdir(parents=True, exist_ok=True)
res = subprocess.run(["uv", "run", "pytest", "--cov=src"], capture_output=True, text=True, check=False)
(p / "test_execution_log.txt").write_text(res.stdout + res.stderr)
