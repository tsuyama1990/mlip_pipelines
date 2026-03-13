from src.domain_models.config import PolicyConfig
from src.domain_models.dtos import ExplorationStrategy, MaterialFeatures


class AdaptiveExplorationPolicyEngine:
    """Decides the optimal exploration strategy based on material features."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide_policy(self, features: MaterialFeatures) -> ExplorationStrategy:
        # Default strategy
        strategy = ExplorationStrategy(policy_name="Standard", md_mc_ratio=0.0, t_max=self.config.default_t_max_scale * features.melting_point, n_defects=0.0, strain_range=0.0)

        # Rule 1: High Initial Uncertainty -> Cautious Exploration
        if features.initial_gamma_variance > 1.0:
            strategy.policy_name = "Cautious Exploration"
            strategy.t_max = self.config.cautious_t_max_scale * features.melting_point
            return strategy

        # Rule 2: Metal & Multi-component -> High-MC Policy
        if features.band_gap <= 0.1 and len(features.elements) > 1:
            strategy.policy_name = "High-MC Policy"
            strategy.md_mc_ratio = 100.0  # Just an arbitrary positive number per spec
            strategy.t_max = self.config.high_mc_t_max_scale * features.melting_point
            return strategy

        # Rule 3: Insulator -> Defect-Driven Policy
        if features.band_gap > 0.1:
            strategy.policy_name = "Defect-Driven Policy"
            strategy.n_defects = 0.05
            strategy.md_mc_ratio = 0.0
            return strategy

        # Rule 4: Hard material -> Strain-Heavy Policy
        if features.bulk_modulus > 200.0:
            strategy.policy_name = "Strain-Heavy Policy"
            strategy.strain_range = 0.15
            return strategy

        return strategy
