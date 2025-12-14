from omegaconf import DictConfig, OmegaConf
from core.interfaces import AbstractPotential
from potentials.mace_impl import MacePotential
from potentials.linear_ace_impl import LinearACEPotential
from loguru import logger

def load_potential(config: DictConfig) -> AbstractPotential:
    """
    Factory function to instantiate the correct potential based on configuration.
    """
    arch = config.potential.arch
    device = config.get("device", "cpu")

    if arch == "mace":
        model_path = config.potential.mace.model_path
        logger.info(f"Loading MacePotential from {model_path} on {device}")
        return MacePotential(model_path, device=device)

    elif arch == "linear_ace":
        # Extract LJ params
        lj_conf = config.potential.delta_learning.lj_params
        # Convert to dict if it's DictConfig
        if OmegaConf.is_config(lj_conf):
            lj_params = OmegaConf.to_container(lj_conf, resolve=True)
        else:
            lj_params = lj_conf

        model_path = config.potential.linear_ace.get("model_path", "linear_ace.yace")

        # Atomic energies might be in config or empty initially
        atomic_energies = config.potential.get("atomic_energies", {})
        if OmegaConf.is_config(atomic_energies):
             atomic_energies = OmegaConf.to_container(atomic_energies, resolve=True)

        logger.info(f"Loading LinearACEPotential from {model_path} with baseline {lj_params}")

        return LinearACEPotential(
            model_path=model_path,
            atomic_energies=atomic_energies,
            lj_params=lj_params,
            device=device
        )

    else:
        raise ValueError(f"Unknown potential architecture: {arch}")
