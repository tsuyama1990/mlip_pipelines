import re
with open("src/dynamics/eon_driver.py", "r") as f:
    data = f.read()

# Fix UnboundLocalError: cannot access local variable 'atoms_obj' where it is not associated with a value
# In read_coordinates_from_stdin:
#         try:
#             atoms_obj = read(io.StringIO(content), format="extxyz")
#         except Exception as e:
#             sys.stderr.write(f"Failed to parse input stream: {e}\n")
#             sys.exit(100)
# Wait, if sys.exit is mocked, it continues to the next line and throws UnboundLocalError.
# We just need to add a return or ensure we actually exit in tests. But since the tests mock `sys.exit(100)` to not raise, we should actually raise SystemExit in tests, OR just fix the function to return early.
# We can just change `sys.exit(100)` mock in tests to actually raise SystemExit.

with open("tests/unit/test_eon_driver.py", "r") as f:
    test_data = f.read()

# change `monkeypatch.setattr("sys.exit", lambda x: None)` to `monkeypatch.setattr("sys.exit", lambda x: sys.exit(x))`
# Actually, the original tests used `with pytest.raises(SystemExit)`!
# Why did I mock it to None? Ah, because `os.lstat` was failing and I wanted to see the error message.
# I'll just change the mock back to let it raise SystemExit, then capture the message.

test_data = test_data.replace('monkeypatch.setattr("sys.exit", lambda x: None)', '')

with open("tests/unit/test_eon_driver.py", "w") as f:
    f.write(test_data)
