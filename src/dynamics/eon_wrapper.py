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

        resolved_pot_str = ""
        if potential:
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
            if not re.match(r"^[/a-zA-Z0-9_.-]+$", resolved_pot_str) or "\x00" in resolved_pot_str or ".." in resolved_pot_str:
                msg = "Potential path contains invalid characters"
                raise ValueError(msg)

        pot_str = repr(resolved_pot_str) if potential else "None"

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

        driver_content = self.config.eon_driver_template.format(
            executable=shlex.quote(executable),
            threshold=float(self.config.uncertainty_threshold),
            pot_str=pot_str,
        )
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
        for k in self.config.safe_env_keys:
            if k in os.environ:
                val = os.environ[k]
                if k == "PATH":
                    # Validate PATH to ensure it only contains allowed characters and paths
                    paths = val.split(os.pathsep)
                    safe_paths = []
                    for raw_p in paths:
                        clean_p = raw_p.strip()
                        if not clean_p:
                            continue
                        p_obj = Path(clean_p).resolve(strict=False)
                        # Only allow existing absolute paths
                        if p_obj.is_absolute() and p_obj.is_dir():
                            safe_paths.append(str(p_obj))
                    env[k] = os.pathsep.join(safe_paths)
                elif re.match(r"^[/a-zA-Z0-9_.-]+$", val):
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
            cmd: list[str] = [str(eon_bin_path)]

            # We use check=False to capture return code 100 gracefully
            # Create a minimal safe environment whitelist to prevent sensitive credential leaks
            env = self._build_safe_env()

            # Safely invoke EON client using direct list execution through subprocess (shell=False)
            res: subprocess.CompletedProcess[bytes] = subprocess.run(  # noqa: S603
                cmd,
                cwd=str(resolved_work_dir.absolute()),
                capture_output=True,
                shell=False,
                env=env,
                check=False,
            )

            if res.returncode == 100:
                return {
                    "halted": True,
                    "dump_file": resolved_work_dir / "bad_structure.cfg",
                    "is_kmc": True,
                }
            if res.returncode != 0:
                # Other error
                import logging

                logging.error(f"EON client failed with return code {res.returncode}")
                msg = f"EON client failed with return code {res.returncode}"
                raise RuntimeError(msg)

        except FileNotFoundError as e:
            # Re-raise the FileNotFoundError since mocks are not allowed
            msg = "EON client executable not found."
            raise RuntimeError(msg) from e

        return {"halted": False}
