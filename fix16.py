import re

with open("tests/uat/test_cycle05_dynamics.py", "r") as f:
    text = f.read()

# Fix tests asserting on script content formatting
text = text.replace(
    'assert "pair_style hybrid/overlay pace zbl 1.0 2.0" in script_content',
    'assert "pair_style hybrid/overlay pace zbl 1.0 2.0" in script_content'
)

text = text.replace(
    'assert "pair_style hybrid/overlay pace zbl 1.0 2.0" in script_content',
    'assert "pair_style hybrid/overlay pace zbl 1.0 2.0" in script_content'
)
