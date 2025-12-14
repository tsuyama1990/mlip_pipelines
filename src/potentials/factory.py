from omegaconf import DictConfig, OmegaConf
from core.interfaces import AbstractPotential
from potentials.mace_impl import MacePotential
from potentials.pyace_impl import PyACEPotential
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

    elif arch == "pyace":
        # Extract Delta Learning settings
        delta_conf = config.potential.pyace.get("delta_learning", {})

        if not delta_conf.get("enabled", False):
            # MACE uses Direct, PyACE uses Delta. This is a constraint check.
            logger.warning("PyACE enabled but delta_learning is disabled. Enforcing Delta Learning implies baseline subtraction/addition logic will be active if params are present.")

        lj_conf = delta_conf.get("lj_params", {})
        if OmegaConf.is_config(lj_conf):
            lj_params = OmegaConf.to_container(lj_conf, resolve=True)
        else:
            lj_params = lj_conf

        model_path = config.potential.pyace.get("model_path", "pyace.yace")

        # Atomic energies
        atomic_energies = config.potential.get("atomic_energies", {})
        if OmegaConf.is_config(atomic_energies):
             atomic_energies = OmegaConf.to_container(atomic_energies, resolve=True)

        logger.info(f"Loading PyACEPotential from {model_path} with baseline {lj_params}")

        return PyACEPotential(
            model_path=model_path,
            atomic_energies=atomic_energies,
            lj_params=lj_params,
            device=device
        )

    else:
        raise ValueError(f"Unknown potential architecture: {arch}")
