with open("src/core/orchestrator.py") as f:
    content = f.read()

# 1. Update _secure_copy_potential
old_secure_copy = """        # Validate paths for traversal characters
        if ".." in src_pot.name or "/" in src_pot.name or "\\\\" in src_pot.name:
            msg = "Path traversal characters detected in destination filename"
            raise ValueError(msg)

        if not src_pot.is_relative_to(tmp_work_dir):
            msg = "Source potential must reside within the tmp working directory"
            raise ValueError(msg)

        if ".." in final_dest.name or "/" in final_dest.name or "\\\\" in final_dest.name:
            msg = "Path traversal characters detected in destination filename"
            raise ValueError(msg)"""

new_secure_copy = """        # Rely strictly on Path operations and strict resolution
        src_resolved = src_pot.resolve(strict=True)
        if not src_resolved.is_relative_to(tmp_work_dir.resolve(strict=True)):
            msg = "Source potential must strictly reside within the tmp working directory"
            raise ValueError(msg)

        final_resolved = final_dest.resolve(strict=False)
        if not final_resolved.is_relative_to(pot_dir.resolve(strict=True)):
            msg = "Destination potential must strictly reside within the potentials directory"
            raise ValueError(msg)"""

content = content.replace(old_secure_copy, new_secure_copy)

# 2. Remove Exception catch in _run_exploration
import re

content = re.sub(r'        except Exception as e:\n            logging\.exception\("An unexpected critical failure occurred during exploration."\)\n            msg = "Critical infrastructure failure during exploration."\n            raise RuntimeError\(msg\) from e', '', content)

# 3. Simplify _manage_directories to use standard atomic ops
old_swap = """    def _swap_directories(self, tmp_work_dir: Path, work_dir: Path) -> None:
        try:
            # Atomic replacement if supported by OS
            os.replace(tmp_work_dir, work_dir)
        except OSError:
            # Fallback for cross-device links
            backup_dir = work_dir.with_name(work_dir.name + "_backup")
            if work_dir.exists():
                shutil.move(str(work_dir), str(backup_dir))
            try:
                shutil.move(str(tmp_work_dir), str(work_dir))
            except Exception as e:
                # Rollback if failure occurs
                if backup_dir.exists():
                    shutil.move(str(backup_dir), str(work_dir))
                msg = f"Failed to atomically move directories: {e}"
                raise RuntimeError(msg) from e
            finally:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)"""

new_swap = """    def _swap_directories(self, tmp_work_dir: Path, work_dir: Path) -> None:
        try:
            shutil.move(str(tmp_work_dir), str(work_dir))
        except Exception as e:
            msg = f"Failed to cleanly move directories: {e}"
            raise RuntimeError(msg) from e"""

content = content.replace(old_swap, new_swap)

with open("src/core/orchestrator.py", "w") as f:
    f.write(content)
