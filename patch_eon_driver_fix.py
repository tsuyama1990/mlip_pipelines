with open("src/dynamics/eon_driver.py") as f:
    data = f.read()

# Replace those newlines cleanly
data = data.replace('\n")\n', '\\n")\n')

with open("src/dynamics/eon_driver.py", "w") as f:
    f.write(data)
