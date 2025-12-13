import sys
from typing import List, Optional
from pathlib import Path
from loguru import logger
from ase import Atoms
from ase.data import atomic_numbers
import omegaconf

# Add external repo to sys.path
# File path: src/core/generators/adapter.py
# parents[0]=generators, parents[1]=core, parents[2]=src
EXTERNAL_PATH = Path(__file__).resolve().parents[2] / "external" / "mlip_struc_gen" / "src"
if str(EXTERNAL_PATH) not in sys.path:
    sys.path.append(str(EXTERNAL_PATH))
    logger.debug(f"Added {EXTERNAL_PATH} to sys.path")

from src.config.settings import GeneratorSettings

# Import from external (wrapped in try-except to handle potential import errors gracefully)
try:
    from nnp_gen.generators.factory import GeneratorFactory
    from nnp_gen.core.config import AlloySystemConfig, PhysicsConstraints
    from nnp_gen.core.interfaces import BaseGenerator
except ImportError as e:
    logger.critical(f"Failed to import from external repository: {e}")
    raise

def validate_elements(elements: List[str]) -> None:
    for elem in elements:
        if elem not in atomic_numbers:
            raise ValueError(f"Invalid element: {elem}")

class ExternalGeneratorAdapter:
    def __init__(self, settings: GeneratorSettings, seed: int = 42):
        self.settings = settings
        self.seed = seed
        self.generator = self._initialize_generator()

    def _initialize_generator(self) -> BaseGenerator:
        """
        Initialize the external generator.
        """
        # Map GeneratorSettings to AlloySystemConfig (or others based on type)
        # For MVP we stick to Alloy as per prompt assumption, but can expand.

        # Construct config object
        # Note: External code expects 'elements' list.
        elements = self.settings.elements
        if not elements:
            elements = [self.settings.target_element]

        validate_elements(elements)

        # Create PhysicsConstraints (using defaults or deriving)
        constraints = PhysicsConstraints()

        # Instantiate AlloySystemConfig
        # We need to map our simple settings to the complex external config
        # Assuming AlloySystemConfig for now.

        external_config = AlloySystemConfig(
            type="alloy",
            elements=elements,
            constraints=constraints,
            supercell_size=[self.settings.supercell_size] * 3,
            vacancy_concentration=self.settings.vacancy_concentration,
            # Defaults for required fields
            composition_mode="random",
            rattle_std=0.01,
            vol_scale_range=[0.95, 1.05],
            n_initial_structures=1
        )

        logger.debug(f"Initialized external config: {external_config}")

        # Use Factory
        return GeneratorFactory.get_generator(external_config, seed=self.seed)

    def generate(self) -> Atoms:
        """
        Generate a structure.
        """
        logger.info("Calling external generator...")
        try:
            # generator.generate() returns List[Atoms]
            # BaseGenerator.generate() -> List[Atoms]
            atoms_list = self.generator.generate()

            if not atoms_list:
                raise RuntimeError("External generator returned empty list.")

            atoms = atoms_list[0] # Take the first one
            logger.info(f"Generated structure: {atoms}")
            return atoms

        except Exception as e:
            logger.error(f"Error during structure generation: {e}")
            raise
