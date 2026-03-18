import re

with open("tests/unit/test_dynamics_engine.py", "r") as f:
    text = f.read()

# I had only replaced in test_resume_missing_executable:
# config = DynamicsConfig(lmp_binary="lmp", project_root=str(Path.cwd()), trusted_directories=[])
# But wait, looking at my previous diff I forgot the lmp_binary.
# The error says: "ValueError: Potential path must be within the project root:"
text = text.replace('config = DynamicsConfig(lmp_binary="lmp", project_root=str(Path.cwd()), trusted_directories=[])', 'config = DynamicsConfig(lmp_binary="lmp", project_root=str(tmp_path), trusted_directories=[])')

with open("tests/unit/test_dynamics_engine.py", "w") as f:
    f.write(text)
