import subprocess
from pathlib import Path
from typing import Any

from src.core import AbstractDynamics
from src.domain_models.config import DynamicsConfig, SystemConfig


class EONWrapper(AbstractDynamics):
    """Manages EON execution and OTF handling."""

    def __init__(self, config: DynamicsConfig, system_config: SystemConfig) -> None:
        self.config = config
        self.system_config = system_config

    def _write_config_ini(self, work_dir: Path) -> None:
        import re
        if not re.match(r"^[a-zA-Z0-9_-]+$", self.config.eon_job):
            msg = f"Invalid EON job string: {self.config.eon_job}"
            raise ValueError(msg)
        if not re.match(r"^[a-zA-Z0-9_-]+$", self.config.eon_min_mode_method):
            msg = f"Invalid EON min_mode_method string: {self.config.eon_min_mode_method}"
            raise ValueError(msg)

        ini_content = self.config.eon_config_template.format(
            eon_job=self.config.eon_job,
            temperature=self.config.temperature,
            eon_min_mode_method=self.config.eon_min_mode_method,
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
            import os

            resolved_pot = Path(os.path.realpath(potential)).resolve(strict=True)
            if hasattr(self.config, "project_root"):
                root = Path(os.path.realpath(self.config.project_root)).resolve(strict=True)
                if not resolved_pot.is_relative_to(root):
                    msg = f"Potential path must be within the project root: {resolved_pot}"
                    raise ValueError(msg)
            resolved_pot_str = str(resolved_pot)

        pot_str = repr(resolved_pot_str) if potential else "None"

        # Ensure executable doesn't break python logic
        import re
        import sys

        executable = sys.executable
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", executable):
            msg = "Invalid python executable path"
            raise ValueError(msg)

        driver_content = self.config.eon_driver_template.format(
            executable=executable,
            threshold=float(self.config.uncertainty_threshold),
            pot_str=pot_str,
        )
        with Path.open(driver_path, "w") as f:
            f.write(driver_content)
        driver_path.chmod(0o755)

    def run_exploration(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:
        """Runs MD or KMC exploration until a halt condition or completion."""
        return self.run_kmc(potential, work_dir)

    def run_kmc(self, potential: Path | None, work_dir: Path) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
        """Runs EON client in the specified working directory."""
        resolved_work_dir = work_dir.resolve(strict=False)

        # Verify that the resolved working directory is within the project root to prevent traversal
        if hasattr(self.config, "project_root"):
            proj_root = Path(self.config.project_root).resolve(strict=True)
            if not resolved_work_dir.is_relative_to(proj_root):
                msg = f"Working directory {resolved_work_dir} is outside the allowed project root."
                raise ValueError(msg)

        resolved_work_dir.mkdir(parents=True, exist_ok=True)
        self._write_config_ini(resolved_work_dir)
        self._write_pace_driver(resolved_work_dir, potential)

        try:
            # We execute 'eonclient'. If it's missing, subprocess raises FileNotFoundError.
            import re
            import shutil
            import sys

            eon_binary: str = self.config.eon_binary

            if not re.match(r"^[a-zA-Z0-9_-]+$", eon_binary):
                msg = f"Invalid EON binary name: {eon_binary}"
                raise ValueError(msg)

            trusted_dirs: list[str] = self.config.trusted_directories.copy()
            trusted_dirs.append(str(Path(sys.prefix) / "bin"))
            if hasattr(self.config, "project_root"):
                trusted_dirs.append(str(Path(self.config.project_root) / "bin"))

            resolved_which: str | None = shutil.which(eon_binary)
            if resolved_which is None:
                msg = "EON client executable not found."
                raise RuntimeError(msg)

            import os

            eon_path: Path = Path(os.path.realpath(resolved_which)).resolve(strict=True)
            if eon_path.is_symlink():
                msg = "EON binary cannot be a symlink."
                raise ValueError(msg)

            if not eon_path.is_file() or not os.access(eon_path, os.X_OK):
                msg = f"EON binary is not an executable file: {eon_path}"
                raise ValueError(msg)

            if eon_path.name != "eonclient":
                msg = f"Resolved EON binary name must be 'eonclient', got '{eon_path.name}'"
                raise ValueError(msg)

            is_trusted = False
            for td in trusted_dirs:
                try:
                    if eon_path.is_relative_to(Path(td).resolve(strict=True)):
                        is_trusted = True
                        break
                except OSError:
                    continue

            if not is_trusted:
                msg = f"Resolved EON binary must reside in a trusted directory: {eon_path}"
                raise ValueError(msg)

            eon_bin = str(eon_path.absolute())

            cmd: list[str] = [eon_bin]

            # We use check=False to capture return code 100 gracefully
            import os

            env: dict[str, str] = os.environ.copy()
            # Clear sensitive/potentially hazardous env vars
            for k in ["LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"]:
                env.pop(k, None)

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
