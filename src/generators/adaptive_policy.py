from typing import Any

from src.domain_models.dtos import ExplorationStrategy


class AdaptivePolicy:
    """Adaptive exploration policy engine to dynamically output parameters."""

    def __init__(
        self,
        material_dna: dict[str, Any],
        predicted_properties: dict[str, Any],
        policy_config: Any = None,
    ) -> None:
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

        # Driven by PolicyConfig without hardcoded if/else trees for logic.
        # We evaluate rules mapped to configs.

        # Rule evaluation helper
        def apply_strategy(prefix: str, t_max_calc: float) -> ExplorationStrategy:
            return ExplorationStrategy(
                policy_type=prefix,
                r_md_mc=getattr(self.policy_config, f"{prefix}_r_md_mc"),
                t_schedule=(300.0, t_max_calc, getattr(self.policy_config, f"{prefix}_steps")),
                n_defects=getattr(self.policy_config, f"{prefix}_defects"),
                strain_range=getattr(self.policy_config, f"{prefix}_strain"),
            )

        # Mapping conditions to configs
        if eg < self.policy_config.metal_eg_threshold and components > 1:
            return apply_strategy("high_mc", tm * self.policy_config.high_mc_t_max_ratio)

        if eg >= self.policy_config.metal_eg_threshold:
            return apply_strategy("defect", tm * self.policy_config.defect_t_max_ratio)

        if b0 > self.policy_config.hard_b0_threshold:
            return apply_strategy("strain", self.policy_config.strain_t_max)

        return apply_strategy("std", self.policy_config.std_t_max)
