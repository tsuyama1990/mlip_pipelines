import re

with open("src/dynamics/dynamics_engine.py", "r") as f:
    text = f.read()

text = text.replace('        tmp_in_file.write(script + "\\n"))', '        tmp_in_file.write(script + "\\n")')

lines = text.split('\n')
for i, line in enumerate(lines):
    if line.strip() == 'f.write(script + "':
        lines[i] = '            f.write(script + "\\n")'

text = '\n'.join(lines)
with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(text)
