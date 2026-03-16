import os
import re
import shutil
import sys
from pathlib import Path

from src.domain_models.config import TrainerConfig


class BinaryResolverMixin:
    """Mixin class providing secure binary resolution logic."""

    config: TrainerConfig

    def _get_trusted_dirs(self) -> list[str]:
        trusted = list(self.config.trusted_directories)
        trusted.append(str(Path(sys.prefix) / "bin"))
        if hasattr(self.config, "project_root"):
            trusted.append(str(Path(self.config.project_root) / "bin"))
        return trusted

    def _validate_binary_properties(
        self, resolved_bin: Path, binary_name: str, trusted_dirs: list[str]
    ) -> None:
        import stat

        # Use os.lstat to avoid TOCTOU if symlinks are swapped
        try:
            st = os.lstat(resolved_bin)
        except OSError as e:
            msg = f"Binary not found or inaccessible: {resolved_bin}"
            raise ValueError(msg) from e

        if stat.S_ISLNK(st.st_mode):
            msg = f"Binary path resolves to a symlink unexpectedly: {resolved_bin}"
            raise ValueError(msg)

        if not stat.S_ISREG(st.st_mode) or not os.access(resolved_bin, os.X_OK):
            msg = f"Binary is not an executable file: {resolved_bin}"
            raise ValueError(msg)

        if resolved_bin.name != binary_name:
            msg = f"Resolved binary name must be '{binary_name}', got '{resolved_bin.name}'"
            raise ValueError(msg)

        is_trusted = False
        for trusted_path in trusted_dirs:
            try:
                resolved_trusted = Path(trusted_path).resolve(strict=True)
                if resolved_bin.is_relative_to(resolved_trusted):
                    is_trusted = True
                    break
            except OSError:
                continue
        if not is_trusted:
            msg = f"Resolved binary must reside securely in a trusted directory: {resolved_bin}"
            raise ValueError(msg)

    def _resolve_absolute_binary(
        self, binary_setting: str, binary_name: str, trusted_dirs: list[str]
    ) -> str:
        # Detect symlinks before strict resolution
        if Path(binary_setting).is_symlink():
            msg = "Absolute binary path cannot be a symlink."
            raise ValueError(msg)

        canonical_bin = Path(binary_setting).resolve(strict=True)

        self._validate_binary_properties(canonical_bin, binary_name, trusted_dirs)
        self._verify_hash(canonical_bin, binary_name)
        return str(canonical_bin)

    def _resolve_relative_binary(
        self, binary_setting: str, binary_name: str, trusted_dirs: list[str]
    ) -> str:
        if not re.match(r"^[-a-zA-Z0-9_.]+$", binary_setting):
            msg = f"Invalid binary name: {binary_setting}"
            raise ValueError(msg)

        resolved_which = shutil.which(binary_setting)
        if resolved_which is None:
            return binary_setting

        # Detect symlinks before strict resolution
        if Path(resolved_which).is_symlink():
            msg = "Resolved relative binary path cannot be a symlink."
            raise ValueError(msg)

        canonical_bin = Path(resolved_which).resolve(strict=True)

        self._validate_binary_properties(canonical_bin, binary_name, trusted_dirs)
        self._verify_hash(canonical_bin, binary_name)
        return str(canonical_bin)

    def _verify_hash(self, resolved_bin: Path, binary_name: str) -> None:
        expected_hash = self.config.binary_hashes.get(binary_name)
        if expected_hash:
            import hashlib

            h = hashlib.sha256()
            with Path.open(resolved_bin, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                msg = f"Executable hash mismatch for {resolved_bin}"
                raise ValueError(msg)

    def _resolve_binary_path(self, binary_setting: str, binary_name: str) -> str:
        trusted_dirs = self._get_trusted_dirs()
        if Path(binary_setting).is_absolute():
            return self._resolve_absolute_binary(binary_setting, binary_name, trusted_dirs)
        return self._resolve_relative_binary(binary_setting, binary_name, trusted_dirs)
