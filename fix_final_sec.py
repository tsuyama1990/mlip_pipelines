with open("src/dynamics/security_utils.py") as f:
    content = f.read()

# Remove the regex block
old_regex = """    if not re.match(r"^[a-zA-Z0-9_-]+$", executable_name):
        msg = f"Invalid binary name: {executable_name}"
        raise ValueError(msg)"""
content = content.replace(old_regex, "")

with open("src/dynamics/security_utils.py", "w") as f:
    f.write(content)
