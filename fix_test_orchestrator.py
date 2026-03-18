from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
# We must clean up `/home/jules/potentials` if it exists since tests are sharing it
content = content.replace('pot_dir.mkdir(parents=True)', 'if pot_dir.exists():\n            import shutil\n            shutil.rmtree(pot_dir)\n        pot_dir.mkdir(parents=True)')
content = content.replace('pot_dir.mkdir()', 'if pot_dir.exists():\n            import shutil\n            shutil.rmtree(pot_dir)\n        pot_dir.mkdir(parents=True)')
content = content.replace('tmp_work_dir.mkdir(parents=True)', 'if tmp_work_dir.exists():\n            import shutil\n            shutil.rmtree(tmp_work_dir)\n        tmp_work_dir.mkdir(parents=True)')

# Fix `NameError: name 'tmp_path' is not defined`
content = content.replace('project_root = tmp_path', 'project_root = "/home/jules"')
Path("tests/unit/test_orchestrator.py").write_text(content)
