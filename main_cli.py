import sys
import time
from datetime import datetime
from pathlib import Path
from loguru import logger
import torch
import numpy as np
from ase import Atoms

# Add src to path if running as script
sys.path.append(str(Path(__file__).parent))

from src.config.settings import Settings
from src.core.utils.logging import setup_logging
from src.core.utils.io import save_results
from src.core.generators.adapter import ExternalGeneratorAdapter
from src.core.calculators.mace_factory import get_mace_calculator
from src.core.engines.relaxer import StructureRelaxer

def main():
    # 1. Initialize Settings
    # We can use CLI args here via pydantic-settings source or manual overrides
    # For MVP, we load from .env or defaults
    try:
        settings = Settings()
    except Exception as e:
        print(f"Failed to load settings: {e}")
        return

    # 2. Prepare Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    output_dir = settings.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Setup Logging
    setup_logging(output_dir)
    logger.info(f"Starting mlip-struc-gen-local pipeline. Run ID: {run_id}")
    logger.info(f"Settings: {settings.model_dump()}")

    try:
        # 4. Structure Generation
        logger.info("Step 1: Generating Initial Structure")
        adapter = ExternalGeneratorAdapter(settings.generator)
        atoms = adapter.generate()

        # Perturb slightly to ensure forces are non-zero (optional, good for testing)
        atoms.rattle(stdev=0.05)
        logger.info("Applied rattle(0.05) to initial structure.")

        # 5. Load Calculator
        logger.info("Step 2: Loading MACE Model")
        calc = get_mace_calculator(settings.mace)
        atoms.calc = calc

        # 6. Run Relaxation
        logger.info("Step 3: Running Relaxation")
        relaxer = StructureRelaxer(settings)
        result = relaxer.run(atoms, run_id)

        # 7. Save Results
        logger.info("Step 4: Saving Results")
        save_results(result, settings, output_dir)

        logger.success("Pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
