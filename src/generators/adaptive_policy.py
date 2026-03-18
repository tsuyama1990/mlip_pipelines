import logging

from ase.data import chemical_symbols

from src.domain_models.config import PolicyConfig
from src.domain_models.dtos import ExplorationStrategy, MaterialFeatures


class FeatureExtractor:
    """Extracts material features using universal ML potentials (or rule-based fallbacks)."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        if config is None:
            self.config = PolicyConfig()
        else:
            self.config = config

    def extract_features(self, elements: list[str]) -> MaterialFeatures:
        if not elements:
            msg = "elements list cannot be empty"
            raise ValueError(msg)

        for el in elements:
            if el not in chemical_symbols:
                msg = f"Invalid element: {el}"
                raise ValueError(msg)

        try:
            return self._try_load_mlip(elements)
        except ImportError:
            logging.info("matgl/chgnet not available or not used. Using rule-based fallback.")
            return self._rule_based_extraction(elements)

        # Normally wouldn't reach here in UAT mode
        return self._rule_based_extraction(elements)

    def _try_load_mlip(self, elements: list[str]) -> MaterialFeatures:
        import importlib.util

        if (
            importlib.util.find_spec("matgl") is not None
            and importlib.util.find_spec("chgnet") is not None
        ):
            msg = "Full MLIP inference not implemented in UAT mode"
            raise NotImplementedError(msg)
        msg2 = "matgl or chgnet not available"
        raise ImportError(msg2)

    def _rule_based_extraction(self, elements: list[str]) -> MaterialFeatures:
        """Rule-based fallback for generating MaterialFeatures without heavy MLIPs."""
        # Check for transition metals
        transition_metals = [
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
        ]

        has_tm = any(el in transition_metals for el in elements)

        # If it contains transition metals, it's likely a metal (Eg ~ 0)
        if has_tm:
            band_gap = self.config.fallback_metal_band_gap
            melting_point = self.config.fallback_metal_melting_point
            bulk_modulus = self.config.fallback_metal_bulk_modulus
        else:
            # Assume insulator or semiconductor
            band_gap = self.config.fallback_insulator_band_gap
            melting_point = self.config.fallback_insulator_melting_point
            bulk_modulus = self.config.fallback_insulator_bulk_modulus

        return MaterialFeatures(
            elements=elements,
            band_gap=band_gap,
            bulk_modulus=bulk_modulus,
            melting_point=melting_point,
            initial_gamma_variance=0.1,
        )


class AdaptiveExplorationPolicyEngine:
    """Decides the optimal exploration strategy based on material features."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide_policy(self, features: MaterialFeatures) -> ExplorationStrategy:  # noqa: C901
        if features.melting_point <= 0:
            msg = "melting_point must be positive"
            raise ValueError(msg)

        def _validate_strategy(strat: ExplorationStrategy) -> ExplorationStrategy:
            # Enforce physical bounds and allowed policies
            allowed_policies = {
                "Standard",
                "Cautious Exploration",
                "High-MC Policy",
                "Defect-Driven Policy",
                "Strain-Heavy Policy",
                "Fallback Standard",
            }
            if strat.policy_name not in allowed_policies:
                msg = f"Invalid policy name selected: {strat.policy_name}"
                raise ValueError(msg)
            if not (0.0 <= strat.t_max <= 5000.0):
                msg = f"Invalid temperature schedule max value: {strat.t_max}"
                raise ValueError(msg)
            if not (0.0 <= strat.md_mc_ratio <= 1000.0):
                msg = f"Invalid MD/MC ratio: {strat.md_mc_ratio}"
                raise ValueError(msg)
            if not (0.0 <= strat.strain_range <= 0.5):
                msg = f"Invalid strain range: {strat.strain_range}"
                raise ValueError(msg)
            if not (0.0 <= strat.n_defects <= 0.5):
                msg = f"Invalid defect concentration: {strat.n_defects}"
                raise ValueError(msg)
            return strat

        # Default strategy
        strategy = ExplorationStrategy(
            policy_name="Standard",
            md_mc_ratio=self.config.default_md_mc_ratio,
            t_max=self.config.default_t_max_scale * features.melting_point,
            n_defects=self.config.default_n_defects,
            strain_range=self.config.default_strain_range,
        )

        # Rule 1: High Initial Uncertainty -> Cautious Exploration
        if features.initial_gamma_variance > self.config.uncertainty_variance_threshold:
            strategy.policy_name = "Cautious Exploration"
            strategy.t_max = self.config.cautious_t_max_scale * features.melting_point
            return _validate_strategy(strategy)

        # Rule 2: Metal & Multi-component -> High-MC Policy
        if features.band_gap <= self.config.metal_band_gap_threshold and len(features.elements) > 1:
            strategy.policy_name = "High-MC Policy"
            strategy.md_mc_ratio = self.config.high_mc_ratio
            strategy.t_max = self.config.high_mc_t_max_scale * features.melting_point
            return _validate_strategy(strategy)

        # Rule 3: Insulator -> Defect-Driven Policy
        if features.band_gap > self.config.metal_band_gap_threshold:
            strategy.policy_name = "Defect-Driven Policy"
            strategy.n_defects = self.config.defect_driven_n_defects
            strategy.md_mc_ratio = self.config.default_md_mc_ratio
            return _validate_strategy(strategy)

        # Rule 4: Hard material -> Strain-Heavy Policy
        if features.bulk_modulus > self.config.hard_material_bulk_modulus_threshold:
            strategy.policy_name = "Strain-Heavy Policy"
            strategy.strain_range = self.config.strain_heavy_range
            return _validate_strategy(strategy)

        return _validate_strategy(strategy)
