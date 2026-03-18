import re

with open("tests/uat/test_cycle05_dynamics.py", "r") as f:
    text = f.read()

# Replace assert "pair_style hybrid/overlay pace zbl 1.0 2.0"
text = text.replace(
    'assert "pair_style hybrid/overlay pace zbl 1.0 2.0" in script_content',
    'assert "pair_style hybrid/overlay pace zbl 1.0 2.0" in script_content'
)

with open("tests/uat/test_cycle05_dynamics.py", "w") as f:
    f.write(text)
