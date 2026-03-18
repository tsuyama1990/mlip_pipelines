import re

with open("src/trainers/finetune_manager.py", "r") as f:
    text = f.read()

text = text.replace(
    "os.replace(tmp_path_str, str(train_xyz))", "Path(tmp_path_str).replace(str(train_xyz))"
)
with open("src/trainers/finetune_manager.py", "w") as f:
    f.write(text)

with open("src/trainers/ace_trainer.py", "r") as f:
    text = f.read()

text = text.replace(
    """        if self.config.baseline_potential.lower() not in allowed_baselines:
            if not re.match(r"^[a-zA-Z0-9_-]+$", self.config.baseline_potential):
                msg = f"Invalid baseline potential format: {self.config.baseline_potential}"
                raise ValueError(msg)""",
    """        if self.config.baseline_potential.lower() not in allowed_baselines and not re.match(r"^[a-zA-Z0-9_-]+$", self.config.baseline_potential):
            msg = f"Invalid baseline potential format: {self.config.baseline_potential}"
            raise ValueError(msg)""",
)

with open("src/trainers/ace_trainer.py", "w") as f:
    f.write(text)

with open("src/core/orchestrator.py", "r") as f:
    text = f.read()

text = text.replace("st = os.stat(resolved_src)", "st = resolved_src.stat()")

with open("src/core/orchestrator.py", "w") as f:
    f.write(text)
