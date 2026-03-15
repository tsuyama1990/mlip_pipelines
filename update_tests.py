import re

with open('src/domain_models/config.py', 'r') as f:
    content = f.read()

old_logic = """            # Existence check for custom models
            if not os.path.exists(self.mace_model_path) or not os.path.isfile(self.mace_model_path):
                msg = f"Model file does not exist or is not a file: {self.mace_model_path}"
                raise ValueError(msg)"""

new_logic = """            # Existence check for custom models
            import os
            import stat
            try:
                st = os.stat(self.mace_model_path)
                if not stat.S_ISREG(st.st_mode):
                    msg = f"Model file does not exist or is not a file: {self.mace_model_path}"
                    raise ValueError(msg)
            except OSError as e:
                msg = f"Model file does not exist or is not a file: {self.mace_model_path}"
                raise ValueError(msg) from e"""

content = content.replace(old_logic, new_logic)

with open('src/domain_models/config.py', 'w') as f:
    f.write(content)
