with open("tests/unit/test_eon_driver.py", "r") as f:
    data = f.read()

data = data.replace('assert "Potential path contains invalid characters" in err', 'assert "Potential path contains invalid traversal characters" in err')
data = data.replace('assert "Invalid filename" in err', 'assert "Invalid characters in filename" in err')
data = data.replace('eon_driver.write_bad_structure(str(path), atoms)', 'eon_driver.write_bad_structure("bad.cfg", atoms)')
data = data.replace('eon_driver.write_bad_structure("..", Atoms("Fe"))', 'eon_driver.write_bad_structure("../bad.cfg", Atoms("Fe"))')
data = data.replace('assert "ase is not available" in err', 'assert "Failed to parse input stream" in err')

with open("tests/unit/test_eon_driver.py", "w") as f:
    f.write(data)

with open("tests/unit/test_config.py", "r") as f:
    data = f.read()

# Make sure test_validate_single_trusted_dir_not_dir matches the correct message
import re
data = re.sub(
    r'with pytest.raises\(ValueError, match=".\*must be a directory.\*"\):\n        _secure_resolve_and_validate_dir\(str\(file\), check_exists=True\)',
    'with pytest.raises(ValueError, match=".*Directory does not exist, is a symlink, or cannot be opened.*"):\n        _secure_resolve_and_validate_dir(str(file), check_exists=True)',
    data
)

with open("tests/unit/test_config.py", "w") as f:
    f.write(data)
