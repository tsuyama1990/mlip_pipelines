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
        if not resolved_bin.is_file() or not os.access(resolved_bin, os.X_OK):
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
            msg = f"Resolved binary must reside in a trusted directory: {resolved_bin}"
            raise ValueError(msg)

    def _resolve_absolute_binary(
        self, binary_setting: str, binary_name: str, trusted_dirs: list[str]
    ) -> str:
        canonical_bin = Path(binary_setting).resolve(strict=True)

        in_trusted_dir = False
        for td in trusted_dirs:
            try:
                canonical_td = Path(td).resolve(strict=True)
                if str(canonical_bin).startswith(str(canonical_td)):
                    in_trusted_dir = True
                    break
            except (OSError, ValueError):
                continue

        if not in_trusted_dir:
            msg = "Resolved binary path is not securely within trusted directories."
            raise ValueError(msg)

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

        canonical_bin = Path(resolved_which).resolve(strict=True)

        in_trusted_dir = False
        for td in trusted_dirs:
            try:
                canonical_td = Path(td).resolve(strict=True)
                if str(canonical_bin).startswith(str(canonical_td)):
                    in_trusted_dir = True
                    break
            except (OSError, ValueError):
                continue

        if not in_trusted_dir:
            msg = "Resolved binary path is not securely within trusted directories."
            raise ValueError(msg)

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
