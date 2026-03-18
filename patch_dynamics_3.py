from pathlib import Path

content = Path("src/dynamics/dynamics_engine.py").read_text()

# Safely replace everything after `def extract_high_gamma_structures` with the new logic
new_logic = """    def extract_high_gamma_structures(self, dump_file: Path, threshold: float) -> list[Atoms]:
        \"\"\"Extracts structures with high gamma from LAMMPS dump using O(N) memory streaming.\"\"\"
        if not dump_file.exists():
            msg = f"Dump file not found: {dump_file}"
            raise FileNotFoundError(msg)

        import numpy as np
        from ase.io import iread

        high_gamma: list[Atoms] = []
        last_frame = None

        try:
            # Stream frames one by one instead of loading all into memory
            for atoms in iread(str(dump_file), index=":", format="lammps-dump-text"):  # type: ignore[no-untyped-call]
                last_frame = atoms
                if "c_pace_gamma" in atoms.arrays:
                    gamma_array = atoms.arrays["c_pace_gamma"]
                    if np.max(gamma_array) > threshold:
                        self._reservoir_sample(high_gamma, atoms)
        except StopIteration:
            pass

        if not last_frame:
            msg = f"No structures read from dump file: {dump_file}"
            raise ValueError(msg)

        if not high_gamma:
            return [last_frame]

        return high_gamma

    def _reservoir_sample(self, reservoir: list[Atoms], new_item: Atoms) -> None:
        if len(reservoir) < 100:
            reservoir.append(new_item)
        else:
            import random
            j = random.randint(0, len(reservoir))
            if j < 100:
                reservoir[j] = new_item"""

# Split before extract_high_gamma
idx = content.find("    def extract_high_gamma_structures")
content = content[:idx] + new_logic + "\n"

Path("src/dynamics/dynamics_engine.py").write_text(content)
