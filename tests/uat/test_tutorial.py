import subprocess
import sys
from pathlib import Path


def test_marimo_tutorial(tmp_path):
    # Tests that the tutorial runs headlessly without errors
    tutorial_path = Path("tutorials/uat_and_tutorial.py")
    assert tutorial_path.exists()

    # Run the script directly since it's a valid python script via marimo
    res = subprocess.run([sys.executable, str(tutorial_path)], capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
