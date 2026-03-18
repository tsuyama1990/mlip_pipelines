import re
from pathlib import Path

content = Path("src/dynamics/dynamics_engine.py").read_text()

# Replace `ase.io.read` with `ase.io.iread` for OOM safety.
new_extract_logic = """    def extract_high_gamma_structures(self, dump_file: Path, threshold: float) -> list[Atoms]:
        \"\"\"Extracts structures with high gamma from LAMMPS dump using O(N) memory streaming.\"\"\"
        if not dump_file.exists():
            msg = f"Dump file not found: {dump_file}"
            raise FileNotFoundError(msg)

        from ase.io import iread
        import numpy as np

        high_gamma = []
        last_frame = None

        try:
            # Stream frames one by one instead of loading all into memory
            for atoms in iread(str(dump_file), index=":", format="lammps-dump-text"):  # type: ignore[no-untyped-call]
                last_frame = atoms
                if "c_pace_gamma" in atoms.arrays:
                    gamma_array = atoms.arrays["c_pace_gamma"]
                    if np.max(gamma_array) > threshold:
                        # Reservoir sampling could be used here if we expect millions of high-gamma frames.
                        # However, since high-gamma frames immediately halt the simulation, there should only be a few.
                        # Keeping them in memory is safe, but we cap at 100 to prevent edge-case OOM.
                        if len(high_gamma) < 100:
                            high_gamma.append(atoms)
                        else:
                            # Reservoir sampling: replace a random element with decreasing probability
                            import random
                            j = random.randint(0, len(high_gamma))
                            if j < 100:
                                high_gamma[j] = atoms
        except StopIteration:
            pass

        if not last_frame:
            msg = f"No structures read from dump file: {dump_file}"
            raise ValueError(msg)

        if not high_gamma:
            return [last_frame]

        return high_gamma"""

content = re.sub(
    r'    def extract_high_gamma_structures.*?(?=\Z|\n\n)',
    new_extract_logic,
    content,
    flags=re.DOTALL
)

Path("src/dynamics/dynamics_engine.py").write_text(content)
