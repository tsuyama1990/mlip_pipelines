import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.core import AbstractDynamics
from src.domain_models.config import DynamicsConfig, SystemConfig
from src.dynamics.security_utils import validate_executable_path, validate_filename


class EONWrapper(AbstractDynamics):
    """Manages EON execution and OTF handling."""

    def __init__(self, config: DynamicsConfig, system_config: SystemConfig) -> None:
        self.config = config
        self.system_config = system_config

    def _write_config_ini(self, work_dir: Path) -> None:
        validate_filename(self.config.eon_job)
        validate_filename(self.config.eon_min_mode_method)

        ini_content = self.config.eon_config_template.format(
            eon_job=shlex.quote(self.config.eon_job),
            temperature=self.config.temperature,
            eon_min_mode_method=shlex.quote(self.config.eon_min_mode_method),
        )
        with Path.open(work_dir / "config.ini", "w") as f:
            f.write(ini_content)

    def _write_pace_driver(self, work_dir: Path, potential: Path | None) -> None:
        pots_dir = work_dir / "potentials"
        pots_dir.mkdir(parents=True, exist_ok=True)
        driver_path = pots_dir / "pace_driver.py"

        # the pace_driver should be executable by python

        # Security: strictly validate the potential string before formatting it into the python template

        resolved_pot_str = "None"
        if potential:
            validate_filename(potential.name)
            resolved_pot = Path(os.path.normpath(os.path.realpath(potential))).resolve(strict=True)
            if self.config.project_root:
                root = Path(os.path.normpath(os.path.realpath(self.config.project_root))).resolve(
                    strict=True
                )
                if not resolved_pot.is_relative_to(root):
                    msg = f"Potential path must be within the project root: {resolved_pot}"
                    raise ValueError(msg)
            # Ensure the potential string itself doesn't contain injected template syntax
            resolved_pot_str = str(resolved_pot)
            # Stricter checks for potential path characters
            if (
                not re.match(r"^[/a-zA-Z0-9_.-]+$", resolved_pot_str)
                or "\x00" in resolved_pot_str
                or ".." in resolved_pot_str
            ):
                msg = "Potential path contains invalid characters"
                raise ValueError(msg)

        # Ensure executable doesn't break python logic

        # Secure executable validation
        executable = os.path.realpath(sys.executable)
        exec_path = Path(executable).resolve(strict=False)
        if not exec_path.is_file():
            msg = "Invalid python executable path"
            raise ValueError(msg)
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", executable) or ".." in executable:
            msg = "Invalid python executable path"
            raise ValueError(msg)

        if not isinstance(self.config.uncertainty_threshold, (int, float)):
            msg = "Invalid threshold type"
            raise TypeError(msg)

        # Build a safe shell script wrapper that calls our static python module.
        # This completely removes the need to generate executable python code from templates.

        # We find the static driver location dynamically to make sure it's correct
        static_driver = Path(__file__).parent / "eon_driver.py"
        static_driver = static_driver.resolve(strict=True)

        driver_content = f"""#!{executable}
import subprocess
import sys
import os

cmd = [
    {executable!r},
    {str(static_driver)!r},
    "--threshold", {str(self.config.uncertainty_threshold)!r},
    "--potential", {resolved_pot_str!r},
    "--default_element", {self.system_config.elements[0]!r},
    "--default_cell", {str(self.config.lattice_size)!r}
]

env = os.environ.copy()
# EON sends atoms via stdin, which we must pipe exactly
res = subprocess.run(cmd, input=sys.stdin.read(), text=True, capture_output=True, env=env)
sys.stdout.write(res.stdout)
sys.stderr.write(res.stderr)
sys.exit(res.returncode)
"""
        with Path.open(driver_path, "w") as f:
            f.write(driver_content)
        driver_path.chmod(0o755)

    def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:
        """Runs MD or KMC exploration until a halt condition or completion."""
        return self.run_kmc(potential, work_dir)

    def _validate_work_dir(self, work_dir: Path) -> Path:
        """Validates and resolves the working directory."""
        work_dir.mkdir(parents=True, exist_ok=True)
        resolved_work_dir = work_dir.resolve(strict=True)

        # Verify that the resolved working directory is within the project root to prevent traversal
        if self.config.project_root:
            proj_root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_work_dir.is_relative_to(proj_root):
                msg = f"Working directory {resolved_work_dir} is outside the allowed project root."
                raise ValueError(msg)
        return resolved_work_dir

    def _get_validated_eon_bin(self) -> Path:
        """Resolves and validates the EON binary path."""
        project_root = self.config.project_root
        try:
            eon_bin = validate_executable_path(
                self.config.eon_binary,
                self.config.trusted_directories,
                project_root=str(project_root) if project_root else None,
                expected_hash=self.config.binary_hashes.get(self.config.eon_binary),
            )
        except RuntimeError as e:
            msg = "EON client executable not found."
            raise RuntimeError(msg) from e

        # Strictly validate the resolved EON binary path before execution
        eon_bin_path = Path(os.path.realpath(eon_bin)).resolve(strict=True)
        if not eon_bin_path.is_file():
            msg = "EON binary is not a valid file."
            raise ValueError(msg)
        if not os.access(eon_bin_path, os.X_OK):
            msg = "EON binary is not executable."
            raise ValueError(msg)

        # Verify the binary is within trusted directories
        is_trusted = False
        for td in self.config.trusted_directories:
            td_path = Path(os.path.realpath(td)).resolve(strict=False)
            if eon_bin_path.is_relative_to(td_path):
                is_trusted = True
                break

        if not is_trusted:
            msg = "EON binary is not within trusted directories."
            raise ValueError(msg)

        return eon_bin_path

    def _build_safe_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        safe_keys = ("PATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")
        for k in safe_keys:
            if k in os.environ:
                val: str = os.environ[k]
                if not isinstance(val, str):
                    continue
                if k in ("PATH", "LD_LIBRARY_PATH"):
                    # Validate PATH-like variables to ensure they only contain allowed characters and paths
                    paths: list[str] = val.split(os.pathsep)
                    safe_paths: list[str] = []
                    for raw_p in paths:
                        clean_p: str = raw_p.strip()
                        if not clean_p:
                            continue
                        p_obj: Path = Path(clean_p).resolve(strict=False)
                        # Only allow existing absolute paths
                        if p_obj.is_absolute() and p_obj.is_dir():
                            safe_paths.append(str(p_obj))
                    env[k] = os.pathsep.join(safe_paths)
                elif re.match(r"^[/a-zA-Z0-9_.-:=]+$", val):
                    env[k] = val
        return env

    def run_kmc(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:
        """Runs EON client in the specified working directory."""
        resolved_work_dir = self._validate_work_dir(work_dir)

        self._write_config_ini(resolved_work_dir)
        self._write_pace_driver(resolved_work_dir, potential)

        try:
            # We execute 'eonclient'. If it's missing, subprocess raises FileNotFoundError.
            eon_bin_path = self._get_validated_eon_bin()

            if not eon_bin_path.is_absolute() or not eon_bin_path.is_file():
                msg = "Invalid resolved EON binary path"
                raise ValueError(msg)

            cmd: list[str] = [str(eon_bin_path)]

            # Create a minimal safe environment whitelist to prevent sensitive credential leaks
            env = self._build_safe_env()

            # Execute via Popen to properly manage the process lifecycle

            with subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(resolved_work_dir.absolute()),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                env=env,
            ) as proc:
                try:
                    out, err = proc.communicate(timeout=3600)  # Maximum 1 hour timeout
                    returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()  # ensure pipes are closed
                    msg = "EON client execution timed out."
                    raise RuntimeError(msg) from None

            if returncode == 100:
                return {
                    "halted": True,
                    "dump_file": resolved_work_dir / "bad_structure.cfg",
                    "is_kmc": True,
                }
            if returncode != 0:
                # Other error
                import logging

                logging.error(f"EON client failed with return code {returncode}")
                msg = f"EON client failed with return code {returncode}"
                raise RuntimeError(msg)

        except FileNotFoundError as e:
            # Re-raise the FileNotFoundError since mocks are not allowed
            msg = "EON client executable not found."
            raise RuntimeError(msg) from e

        return {"halted": False}
