from typing import Any

from src.domain_models.dtos import ExplorationStrategy


class AdaptivePolicy:
    """Adaptive exploration policy engine to dynamically output parameters."""

    def __init__(self, material_dna: dict[str, Any], predicted_properties: dict[str, Any], policy_config: Any = None) -> None:
        self.material_dna = material_dna
        self.predicted_properties = predicted_properties
        # Fallback for existing tests, though we should inject real one
        if policy_config is None:
            from src.domain_models.config import PolicyConfig
            self.policy_config = PolicyConfig()
        else:
            self.policy_config = policy_config

    def generate_strategy(self) -> ExplorationStrategy:
        """Determines the exploration strategy based on material parameters."""
        eg = self.predicted_properties.get("band_gap", 0.0)
        tm = self.predicted_properties.get("melting_point", 1000.0)
        b0 = self.predicted_properties.get("bulk_modulus", 50.0)
        components = len(self.material_dna.get("elements", ["Fe", "Pt"]))

        is_metal = eg < 0.1
        is_hard = b0 > 150.0

        if is_metal and components > 1:
            return ExplorationStrategy(
                policy_type="High-MC",
                r_md_mc=self.policy_config.high_mc_r_md_mc,
                t_schedule=(300.0, tm * self.policy_config.high_mc_t_max_ratio, self.policy_config.high_mc_steps),
                n_defects=self.policy_config.high_mc_defects,
                strain_range=self.policy_config.high_mc_strain,
            )
        if not is_metal:
            return ExplorationStrategy(
                policy_type="Defect-Driven",
                r_md_mc=self.policy_config.defect_r_md_mc,
                t_schedule=(300.0, tm * self.policy_config.defect_t_max_ratio, self.policy_config.defect_steps),
                n_defects=self.policy_config.defect_defects,
                strain_range=self.policy_config.defect_strain,
            )
        if is_hard:
            return ExplorationStrategy(
                policy_type="Strain-Heavy",
                r_md_mc=self.policy_config.strain_r_md_mc,
                t_schedule=(300.0, self.policy_config.strain_t_max, self.policy_config.strain_steps),
                n_defects=self.policy_config.strain_defects,
                strain_range=self.policy_config.strain_strain,
            )
        return ExplorationStrategy(
            policy_type="Standard-MD",
            r_md_mc=self.policy_config.std_r_md_mc,
            t_schedule=(300.0, self.policy_config.std_t_max, self.policy_config.std_steps),
            n_defects=self.policy_config.std_defects,
            strain_range=self.policy_config.std_strain,
        )
