from pathlib import Path

content = Path("tests/unit/test_orchestrator.py").read_text()
content = content.replace('''if pot_dir.exists():
            import shutil
            shutil.rmtree(pot_dir)
        pot_dir.mkdir(parents=True)''',
'''        if pot_dir.exists():
            import shutil
            shutil.rmtree(pot_dir)
        pot_dir.mkdir(parents=True)''')
Path("tests/unit/test_orchestrator.py").write_text(content)
