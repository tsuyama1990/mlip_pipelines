from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
content = content.replace('tmp_path', 'Path("/home/jules").resolve(strict=True)')
Path("tests/unit/test_orchestrator.py").write_text(content)
