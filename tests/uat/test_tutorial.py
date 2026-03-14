import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skip(
    reason="Headless execution of marimo notebooks can cause timeouts or missing dependency errors in CI"
)
def test_marimo_tutorial(tmp_path: Path) -> None:
    # Tests that the tutorial runs headlessly without errors
    tutorial_path = Path("tutorials/uat_and_tutorial.py")
    assert tutorial_path.exists()

    # Run the script directly since it's a valid python script via marimo
    res = subprocess.run(
        [sys.executable, str(tutorial_path)],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        check=False,
    )
    assert res.returncode == 0
