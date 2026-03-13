import subprocess
import sys

res = subprocess.run([sys.executable, "tutorials/UAT_AND_TUTORIAL.py"])
print(res.returncode)
