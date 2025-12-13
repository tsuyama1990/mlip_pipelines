import sys
import time
from pathlib import Path
from tqdm import tqdm
from ase.build import bulk
from ase.io import write
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def main():
    logger.info("Starting Bulk Generation Demo (100 structures)")

    output_dir = Path("output/bulk_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    n_structures = 100
    base_atoms = bulk('Si', 'diamond', a=5.43, cubic=True)

    # Progress Bar
    for i in tqdm(range(n_structures), desc="Generating structures"):
        # Create a variant
        atoms = base_atoms.copy()

        # Apply random strain (volume scaling)
        scale = 0.95 + (1.05 - 0.95) * (i / n_structures)
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)

        # Apply random rattle
        atoms.rattle(stdev=0.02)

        # Save
        filename = output_dir / f"structure_{i:03d}.xyz"
        write(filename, atoms)

        # Simulate some heavy computation
        time.sleep(0.01)

    logger.success(f"Generated {n_structures} structures in {output_dir}")

if __name__ == "__main__":
    main()
