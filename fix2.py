import re

with open("src/trainers/finetune_manager.py") as f:
    text = f.read()

replacement = '''    def _secure_write_xyz(self, train_xyz: Path, structures: list) -> None:
        """Secure atomic write."""
        import tempfile
        import os
        import fcntl
        from ase.io import write

        if train_xyz.exists():
            msg = f"File already exists: {train_xyz}"
            raise FileExistsError(msg)

        # Write to a temporary file first in the same directory, then rename
        fd, tmp_path_str = tempfile.mkstemp(dir=str(train_xyz.parent), prefix=".tmp_finetune_")
        tmp_path = Path(tmp_path_str)
        try:
            # File locking using fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with os.fdopen(fd, "w", encoding="utf-8") as f_out:
                for atoms in structures:
                    write(f_out, atoms, format="extxyz")

            # Atomic rename (POSIX only, but fine for HPC)
            os.replace(tmp_path_str, str(train_xyz))
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            msg = f"Failed to securely write xyz: {e}"
            raise RuntimeError(msg) from e'''

text = re.sub(
    r"    def _secure_write_xyz\(self, train_xyz: Path, structures: list\[Atoms\]\) -> None:[\s\S]*?raise RuntimeError\(msg\) from e",
    replacement,
    text,
)
with open("src/trainers/finetune_manager.py", "w") as f:
    f.write(text)
