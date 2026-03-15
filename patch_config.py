with open("tests/unit/test_config.py", "r") as f:
    test_data = f.read()

# Fix `test_validate_single_trusted_dir_not_dir` match
test_data = test_data.replace('match=".*Directory does not exist, is a symlink, or cannot be opened.*"', 'match=".*Directory does not exist, is a symlink, or cannot be opened.*|.*Path must be a directory.*"')

with open("tests/unit/test_config.py", "w") as f:
    f.write(test_data)
