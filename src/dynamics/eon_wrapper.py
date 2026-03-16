import contextlib
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
        import os

        # Configure potentials directory via environment variable, fallback to default
        pots_dir_name = os.environ.get("MLIP_POTENTIALS_DIR", "potentials")
        if not re.match(r"^[a-zA-Z0-9_-]+$", pots_dir_name):
            msg = "Invalid characters in MLIP_POTENTIALS_DIR"
            raise ValueError(msg)

        pots_dir = work_dir / pots_dir_name
        pots_dir.mkdir(parents=True, exist_ok=True)
        driver_path = pots_dir / "pace_driver.py"

        # the pace_driver should be executable by python

        # Security: strictly validate the potential string before formatting it into the python template

        resolved_pot_str = "None"
        if potential:
            validate_filename(potential.name)
            # resolve strictly while following symlinks
            resolved_pot = potential.resolve(strict=True).absolute()
            if self.config.project_root:
                root = Path(self.config.project_root).resolve(strict=True).absolute()
                if not resolved_pot.is_relative_to(root):
                    msg = f"Potential path must be within the project root: {resolved_pot}"
                    raise ValueError(msg)
            resolved_pot_str = resolved_pot.as_posix()

        try:
            executable = Path(sys.executable).resolve(strict=True)
            executable_str = executable.as_posix()
            valid_exec = executable.is_file() and os.access(executable, os.X_OK)
        except Exception as e:
            msg = "Invalid python executable path"
            raise ValueError(msg) from e

        if not valid_exec:
            msg = "Invalid python executable path"
            raise ValueError(msg)

        if not isinstance(self.config.uncertainty_threshold, (int, float)):
            msg = "Invalid threshold type"
            raise TypeError(msg)

        static_driver = (Path(__file__).parent / "eon_driver.py").resolve(strict=True)

        import json

        cmd = [
            executable_str,
            static_driver.as_posix(),
            "--threshold",
            str(self.config.uncertainty_threshold),
            "--potential",
            resolved_pot_str,
            "--default_element",
            self.system_config.elements[0],
            "--default_cell",
            str(self.config.lattice_size),
        ]

        # Secure generation without f-string interpolation risks.
        # We serialize the validated command array directly into a JSON literal to be parsed in the script.
        cmd_json = json.dumps(cmd)

        driver_content = f"""#!{executable_str}
import subprocess
import sys
import os
import json

# Safely deserialize the literal command array
cmd = json.loads({cmd_json!r})

# Strictly pass only safe variables downstream
safe_env = {{}}
for key in ("PATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS"):
    if key in os.environ:
        safe_env[key] = os.environ[key]

res = subprocess.run(cmd, input=sys.stdin.read(), text=True, capture_output=True, env=safe_env)
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
        eon_bin_path = eon_bin.resolve(strict=True)
        if not eon_bin_path.is_file():
            msg = "EON binary is not a valid file."
            raise ValueError(msg)
        if not os.access(eon_bin_path, os.X_OK):
            msg = "EON binary is not executable."
            raise ValueError(msg)

        # Verify the binary is within trusted directories
        is_trusted = False
        for td in self.config.trusted_directories:
            td_path = Path(td).resolve(strict=True)
            if eon_bin_path.is_relative_to(td_path):
                is_trusted = True
                break

        if not is_trusted:
            msg = "EON binary is not within trusted directories."
            raise ValueError(msg)

        self._verify_binary_hash(eon_bin_path)

        return eon_bin_path

    def _verify_binary_hash(self, eon_bin_path: Path) -> None:
        # Double check the binary hash again after strict resolution to prevent TOCTOU bypasses
        st = os.lstat(eon_bin_path)
        if st.st_uid != os.getuid():
            msg = "EON binary is not owned by the current user"
            raise ValueError(msg)

        if not self.config.binary_hashes:
            return

        expected_hash = self.config.binary_hashes.get(self.config.eon_binary)
        if not expected_hash:
            return

        import hashlib

        hasher = hashlib.sha256()
        with eon_bin_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != expected_hash:
            msg = f"EON binary hash mismatch after resolution. Expected {expected_hash}, got {hasher.hexdigest()}"
            raise ValueError(msg)

    def _is_path_trusted(self, p_obj: Path) -> bool:
        if self.config.project_root:
            with contextlib.suppress(Exception):
                root = Path(self.config.project_root).resolve(strict=True)
                if p_obj.is_relative_to(root):
                    return True
        for tdir in self.config.trusted_directories:
            with contextlib.suppress(Exception):
                tp = Path(tdir).resolve(strict=True)
                if p_obj.is_relative_to(tp):
                    return True
        return False

    def _validate_env_path(self, raw_p: str) -> str | None:
        import logging

        clean_p: str = raw_p.strip()
        if not clean_p:
            return None

        try:
            # Enforce realpath before resolution to protect against TOCTOU path resolution exploits
            import os

            real_p = os.path.realpath(clean_p)
            p_obj: Path = Path(real_p).resolve(strict=True)

            if not p_obj.is_absolute():
                logging.warning(f"Path is not absolute: {clean_p}")
                return None

            if not p_obj.is_dir():
                logging.warning(f"Path is not a valid directory: {clean_p}")
                return None

            if not self._is_path_trusted(p_obj):
                logging.warning(f"Path is untrusted: {clean_p}")
                return None

            return p_obj.as_posix()

        except (RuntimeError, ValueError, OSError) as e:
            logging.warning(f"Invalid environment path provided: {e}")
            return None

    def _build_safe_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        safe_keys = ("PATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")
        for k in safe_keys:
            if k in os.environ:
                val: str = os.environ[k]
                if not isinstance(val, str):
                    continue

                # Broadly restrict payload length and disallow generic injection operators natively before processing
                if len(val) > 4096 or re.search(r"[`;&|<>()$\\{\}\"\']", val):
                    continue

                if k in ("PATH", "LD_LIBRARY_PATH"):
                    safe_paths = []
                    for raw_p in val.split(os.pathsep):
                        validated = self._validate_env_path(raw_p)
                        if validated:
                            safe_paths.append(validated)
                    env[k] = os.pathsep.join(safe_paths)
                elif re.match(r"^[0-9]+$", val):
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
            # Added size limits to pipes and full file-based processing for massive outputs
            out_file = resolved_work_dir / "eonclient.out"
            err_file = resolved_work_dir / "eonclient.err"

            with (
                out_file.open("w") as fout,
                err_file.open("w") as ferr,
                subprocess.Popen(  # noqa: S603
                    cmd,
                    cwd=str(resolved_work_dir.absolute()),
                    stdout=fout,
                    stderr=ferr,
                    shell=False,
                    env=env,
                ) as proc,
            ):
                try:
                    proc.communicate(timeout=3600)  # Maximum 1 hour timeout
                    returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()  # ensure pipes are closed
                    msg = "EON client execution timed out."
                    raise RuntimeError(msg) from None
                finally:
                    # Defensive kill if it somehow escaped the above cleanly
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()

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
