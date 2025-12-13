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
from src.core.exceptions import MLIPPipelineError
from src.core.constants import RATTLE_AMPLITUDE_ANGSTROM

def main():
    # 1. Initialize Settings
    try:
        settings = Settings()
    except Exception as e:
        print(f"Failed to load settings: {e}")
        return

    # Set random seeds for reproducibility
    np.random.seed(settings.random_seed)
    torch.manual_seed(settings.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.random_seed)

    # Create RNG for specific operations
    rng = np.random.default_rng(settings.random_seed)

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
        # Pass full settings or generator settings + seed?
        # Adapter takes GeneratorSettings. We need to update Adapter to take seed or update its interface.
        # Current Adapter takes settings.generator.
        # We will modify Adapter to extract seed from settings if passed, or we pass seed explicitly.
        # Plan says "Update src/core/generators/adapter.py to accept seed".
        # Let's verify how adapter is initialized. It takes `settings.generator`.
        # I'll modify adapter __init__ to accept seed.

        adapter = ExternalGeneratorAdapter(settings.generator, seed=settings.random_seed)
        atoms = adapter.generate()

        # Perturb slightly to ensure forces are non-zero (optional, good for testing)
        atoms.rattle(stdev=RATTLE_AMPLITUDE_ANGSTROM, rng=rng)
        logger.info(f"Applied rattle({RATTLE_AMPLITUDE_ANGSTROM} [Å]) to initial structure using seed {settings.random_seed}.")

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

    except MLIPPipelineError as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected pipeline failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
