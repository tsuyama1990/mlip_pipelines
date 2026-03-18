from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
content = content.replace('pot_dir.mkdir(parents=True)', 'import shutil; shutil.rmtree(pot_dir) if pot_dir.exists() else None; pot_dir.mkdir(parents=True)')
content = content.replace('tmp_work_dir.mkdir(parents=True)', 'import shutil; shutil.rmtree(tmp_work_dir) if tmp_work_dir.exists() else None; tmp_work_dir.mkdir(parents=True)')
content = content.replace('project_root = tmp_path', 'project_root = "/home/jules"')
Path("tests/unit/test_orchestrator.py").write_text(content)
