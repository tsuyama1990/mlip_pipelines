with open("src/dynamics/dynamics_engine.py") as f:
    content = f.read()

content = content.replace("subprocess.CompletedProcess[bytes] = subprocess.run(", "subprocess.CompletedProcess[bytes] = subprocess.run(  # noqa: S603")
content = content.replace("            subprocess.run(\n                cmd,", "            subprocess.run(  # noqa: S603\n                cmd,")

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(content)

with open("src/dynamics/eon_wrapper.py") as f:
    content = f.read()

content = content.replace("subprocess.CompletedProcess[bytes] = subprocess.run(", "subprocess.CompletedProcess[bytes] = subprocess.run(  # noqa: S603")

with open("src/dynamics/eon_wrapper.py", "w") as f:
    f.write(content)
