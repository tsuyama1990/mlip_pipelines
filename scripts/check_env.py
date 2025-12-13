#!/usr/bin/env python3
"""
Task 1: Environment Checker
Verifies pw.x is callable and checks for mpirun if needed.
"""
import shutil
import subprocess
import sys

from loguru import logger


def check_command(cmd: str, args: list[str] = None):
    """Check if a command exists and is runnable."""
    path = shutil.which(cmd)
    if not path:
        logger.error(f"Command '{cmd}' not found in PATH.")
        return False

    logger.info(f"Found '{cmd}' at {path}")

    if args:
        try:
            full_cmd = [cmd] + args
            result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"'{cmd} {' '.join(args)}' ran successfully.")
                # logger.debug(result.stdout)
                return True
            else:
                logger.warning(f"'{cmd}' returned non-zero exit code: {result.returncode}")
                return False
        except Exception as e:
            logger.error(f"Failed to run '{cmd}': {e}")
            return False
    return True

def main():
    logger.info("Checking environment for Quantum ESPRESSO...")

    # Check pw.x (QE main executable)
    # pw.x usually doesn't have --version, but we can try running it with no args or -h
    # Running pw.x without input usually hangs reading from stdin, so we use timeout or expect failure but check existence first.

    if not shutil.which("pw.x"):
        logger.error("CRITICAL: pw.x not found! Please install Quantum ESPRESSO.")
        sys.exit(1)
    else:
        logger.success("pw.x is available.")

    # Check mpirun (Optional but recommended for performance)
    if shutil.which("mpirun"):
        logger.info("mpirun is available.")
        check_command("mpirun", ["--version"])
    else:
        logger.warning("mpirun not found. Calculations will run in serial (slow).")

    logger.success("Environment check passed.")

if __name__ == "__main__":
    main()
