with open("src/dynamics/eon_wrapper.py") as f:
    content = f.read()

content = content.replace('safe_env_keys = ["PATH", "HOME", "USER", "LANG", "LC_ALL"]', 'safe_env_keys = ["PATH"]')

with open("src/dynamics/eon_wrapper.py", "w") as f:
    f.write(content)
