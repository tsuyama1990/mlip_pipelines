import re

with open("tests/unit/test_dynamics_engine.py", "r") as f:
    text = f.read()

# Fix tests asserting on invalid filename format to expect the new regex
text = text.replace(
    'with pytest.raises(ValueError, match="Dump file name contains invalid characters"):',
    'with pytest.raises(ValueError, match="Potential path must be a valid .yace file"):'
)

text = text.replace(
    'with pytest.raises(ValueError, match="Potential path must be a valid .yace file"):\n        engine.run_exploration(potential=pot_file, work_dir=tmp_path)',
    'with pytest.raises(ValueError, match="Potential path must be a valid .yace file"):\n        engine.run_exploration(potential=pot_file, work_dir=tmp_path)'
)

with open("tests/unit/test_dynamics_engine.py", "w") as f:
    f.write(text)
