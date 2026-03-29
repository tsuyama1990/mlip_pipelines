import os
import sys
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.trainers.binary_resolver import BinaryResolverMixin

class MockConfig:
    def __init__(self, trusted_directories=None, binary_hashes=None, project_root=None) -> None:
        self.trusted_directories = trusted_directories or []
        self.binary_hashes = binary_hashes or {}
        if project_root:
            self.project_root = project_root

class DummyResolver(BinaryResolverMixin):
    def __init__(self, config) -> None:
        self.config = config

@pytest.fixture
def mock_bin(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    binary = bin_dir / "test_bin"
    binary.write_text("dummy content")
    binary.chmod(0o755)
    return binary

@pytest.fixture
def trusted_dir(tmp_path):
    d = tmp_path / "trusted"
    d.mkdir()
    return d

def test_get_trusted_dirs_basic(tmp_path):
    trusted_path = str(tmp_path / "trusted")
    config = MockConfig(trusted_directories=[trusted_path])
    resolver = DummyResolver(config)
    trusted = resolver._get_trusted_dirs()

    assert trusted_path in trusted
    assert str(Path(sys.prefix) / "bin") in trusted
    assert not any("project_root" in d for d in trusted)

def test_get_trusted_dirs_with_project_root(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    trusted_path = str(tmp_path / "trusted")
    config = MockConfig(trusted_directories=[trusted_path], project_root=str(project_root))
    resolver = DummyResolver(config)
    trusted = resolver._get_trusted_dirs()

    assert trusted_path in trusted
    assert str(Path(sys.prefix) / "bin") in trusted
    assert str(project_root / "bin") in trusted

def test_validate_binary_properties_happy_path(mock_bin, trusted_dir):
    config = MockConfig(trusted_directories=[str(mock_bin.parent)])
    resolver = DummyResolver(config)
    trusted_dirs = [str(mock_bin.parent)]

    # Should not raise any exception
    resolver._validate_binary_properties(mock_bin, "test_bin", trusted_dirs)

def test_validate_binary_properties_not_found():
    config = MockConfig()
    resolver = DummyResolver(config)
    with pytest.raises(ValueError, match="Binary not found or inaccessible"):
        resolver._validate_binary_properties(Path("/non/existent/bin"), "bin", [])

def test_validate_binary_properties_is_symlink(mock_bin, tmp_path):
    symlink_bin = tmp_path / "symlink_bin"
    symlink_bin.symlink_to(mock_bin)

    config = MockConfig()
    resolver = DummyResolver(config)
    with pytest.raises(ValueError, match="Binary path resolves to a symlink unexpectedly"):
        resolver._validate_binary_properties(symlink_bin, "symlink_bin", [])

def test_validate_binary_properties_not_executable(tmp_path):
    non_exec = tmp_path / "non_exec"
    non_exec.write_text("content")
    non_exec.chmod(0o644)

    config = MockConfig()
    resolver = DummyResolver(config)
    with pytest.raises(ValueError, match="Binary is not an executable file"):
        resolver._validate_binary_properties(non_exec, "non_exec", [])

def test_validate_binary_properties_name_mismatch(mock_bin):
    config = MockConfig()
    resolver = DummyResolver(config)
    with pytest.raises(ValueError, match="Resolved binary name must be 'wrong_name'"):
        resolver._validate_binary_properties(mock_bin, "wrong_name", [])

def test_validate_binary_properties_not_in_trusted_dir(mock_bin, tmp_path):
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    config = MockConfig()
    resolver = DummyResolver(config)
    with pytest.raises(ValueError, match="Resolved binary must reside securely in a trusted directory"):
        resolver._validate_binary_properties(mock_bin, "test_bin", [str(other_dir)])

def test_verify_hash_success(mock_bin):
    content = mock_bin.read_bytes()
    expected_hash = hashlib.sha256(content).hexdigest()
    config = MockConfig(binary_hashes={"test_bin": expected_hash})
    resolver = DummyResolver(config)

    # Should not raise any exception
    resolver._verify_hash(mock_bin, "test_bin")

def test_verify_hash_mismatch(mock_bin):
    config = MockConfig(binary_hashes={"test_bin": "wrong_hash"})
    resolver = DummyResolver(config)

    with pytest.raises(ValueError, match="Executable hash mismatch"):
        resolver._verify_hash(mock_bin, "test_bin")

def test_verify_hash_skipped(mock_bin):
    config = MockConfig(binary_hashes={})
    resolver = DummyResolver(config)

    # Should not raise any exception (skipped)
    resolver._verify_hash(mock_bin, "test_bin")

def test_resolve_absolute_binary_success(mock_bin):
    config = MockConfig(trusted_directories=[str(mock_bin.parent)])
    resolver = DummyResolver(config)

    resolved = resolver._resolve_absolute_binary(str(mock_bin), "test_bin", [str(mock_bin.parent)])
    assert resolved == str(mock_bin.resolve())

def test_resolve_absolute_binary_symlink_fail(mock_bin, tmp_path):
    symlink_bin = tmp_path / "symlink_bin"
    symlink_bin.symlink_to(mock_bin)

    config = MockConfig()
    resolver = DummyResolver(config)
    with pytest.raises(ValueError, match="Absolute binary path cannot be a symlink"):
        resolver._resolve_absolute_binary(str(symlink_bin), "test_bin", [])

def test_resolve_relative_binary_success(mock_bin, monkeypatch):
    config = MockConfig(trusted_directories=[str(mock_bin.parent)])
    resolver = DummyResolver(config)

    import shutil
    monkeypatch.setattr(shutil, "which", lambda x: str(mock_bin))

    resolved = resolver._resolve_relative_binary("test_bin", "test_bin", [str(mock_bin.parent)])
    assert resolved == str(mock_bin.resolve())

def test_resolve_relative_binary_invalid_name():
    config = MockConfig()
    resolver = DummyResolver(config)
    with pytest.raises(ValueError, match="Invalid binary name"):
        resolver._resolve_relative_binary("bin; rm -rf /", "bin", [])

def test_resolve_relative_binary_not_found(monkeypatch):
    config = MockConfig()
    resolver = DummyResolver(config)

    import shutil
    monkeypatch.setattr(shutil, "which", lambda x: None)

    resolved = resolver._resolve_relative_binary("missing_bin", "missing_bin", [])
    assert resolved == "missing_bin"

def test_resolve_relative_binary_symlink_fail(mock_bin, tmp_path, monkeypatch):
    symlink_bin = tmp_path / "symlink_bin"
    symlink_bin.symlink_to(mock_bin)

    config = MockConfig()
    resolver = DummyResolver(config)

    import shutil
    monkeypatch.setattr(shutil, "which", lambda x: str(symlink_bin))

    with pytest.raises(ValueError, match="Resolved relative binary path cannot be a symlink"):
        resolver._resolve_relative_binary("symlink_bin", "symlink_bin", [])

def test_resolve_binary_path_absolute(mock_bin):
    config = MockConfig(trusted_directories=[str(mock_bin.parent)])
    resolver = DummyResolver(config)

    with patch.object(DummyResolver, "_resolve_absolute_binary", return_value="abs_path") as mock_abs:
        res = resolver._resolve_binary_path(str(mock_bin), "test_bin")
        assert res == "abs_path"
        mock_abs.assert_called_once()

def test_resolve_binary_path_relative():
    config = MockConfig(trusted_directories=[])
    resolver = DummyResolver(config)

    with patch.object(DummyResolver, "_resolve_relative_binary", return_value="rel_path") as mock_rel:
        res = resolver._resolve_binary_path("test_bin", "test_bin")
        assert res == "rel_path"
        mock_rel.assert_called_once()
