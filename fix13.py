import re

with open("src/dynamics/dynamics_engine.py", "r") as f:
    text = f.read()

# Fix undefined potential
text = text.replace("        pot_path_str = self._validate_potential_path(potential)\n", "")

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(text)
