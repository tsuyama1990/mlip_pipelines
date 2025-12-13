import torch
from loguru import logger
from ase.calculators.calculator import Calculator
from mace.calculators import mace_mp

from src.config.settings import MACESettings

def get_mace_calculator(settings: MACESettings) -> Calculator:
    """
    Factory to create MACE Calculator.
    """
    model_path = settings.model_path
    device = settings.device
    default_dtype = settings.default_dtype

    logger.info(f"Initializing MACE Calculator: model={model_path}, device={device}, dtype={default_dtype}")

    # Handle mace_mp models
    if model_path in ["small", "medium", "large"]:
        # mace_mp expects 'small', 'medium', 'large' etc.
        # It automatically downloads if needed.
        # We need to ensure we map our settings to mace_mp arguments.
        # mace_mp(model="medium", device="cuda", default_dtype="float64")

        try:
             calc = mace_mp(
                 model=model_path,
                 device=device,
                 default_dtype=default_dtype
             )
             return calc
        except Exception as e:
            logger.error(f"Failed to load MACE MP model: {e}")
            raise
    else:
        # Load from local file
        from mace.calculators import MACECalculator
        try:
            calc = MACECalculator(
                model_paths=model_path,
                device=device,
                default_dtype=default_dtype
            )
            return calc
        except Exception as e:
            logger.error(f"Failed to load local MACE model at {model_path}: {e}")
            raise
