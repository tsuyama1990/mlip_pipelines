from src.domain_models.config import SystemConfig
from src.domain_models.dtos import ExplorationStrategy


class AdaptivePolicy:
    """Engine to determine exploration strategy dynamically."""

    @staticmethod
    def generate_strategy(system_config: SystemConfig) -> ExplorationStrategy:
        """
        Derive exploration strategy based on the elements provided.
        In a full implementation, this queries material DNA and predicted properties.
        """
        # Heuristic rules for materials properties
        # Metals usually have E_g ~ 0, we can use High-MC policy
        metals = {"Fe", "Pt", "Ti", "Cu", "Al", "Ag", "Au", "Ni", "Co"}
        elements_set = set(system_config.elements)

        is_metal = any(e in metals for e in elements_set)

        if is_metal:
            # High-MC Policy
            r_md_mc = 100.0
            t_schedule = [300.0, 600.0, 900.0]
            n_defects = 2
            strain_range = 0.05
        else:
            # Defect-Driven Policy (e.g. for insulators like MgO)
            r_md_mc = 0.0
            t_schedule = [300.0, 500.0]
            n_defects = 10
            strain_range = 0.15

        return ExplorationStrategy(
            r_md_mc=r_md_mc,
            t_schedule=t_schedule,
            n_defects=n_defects,
            strain_range=strain_range,
        )
