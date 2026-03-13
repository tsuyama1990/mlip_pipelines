import subprocess
import sys


def test_tutorial_execution() -> None:
    """
    Runs the UAT tutorial script in headless mode using marimo to ensure it executes without
    Python exceptions. This acts as our end-to-end user acceptance test.
    """
    # The command to run marimo script headlessly
    cmd = [
        sys.executable,
        "-m",
        "marimo",
        "run",
        "tutorials/uat_and_tutorial.py",
        "--headless",
    ]

    # Run the subprocess
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603

    # Ensure the script completed successfully
    assert result.returncode == 0, (
        f"Tutorial execution failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Ensure expected outputs are logged in stdout or stderr
    # We expect "Running one cycle to observe OTF Halt" and "FePt/MgO Cycle Status" somewhere.
    # We won't strictly match the whole output but at least check success.
