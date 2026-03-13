from typing import Any
from src.domain_models.dtos import ExplorationStrategy


class AdaptivePolicy:
    """Adaptive exploration policy engine to dynamically output parameters."""

    def __init__(self, material_dna: dict[str, Any], predicted_properties: dict[str, Any]) -> None:
        self.material_dna = material_dna
        self.predicted_properties = predicted_properties

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
                r_md_mc=100,
                t_schedule=(300.0, tm * 0.8, 20000),
                n_defects=1,
                strain_range=0.05,
            )
        elif not is_metal:
            return ExplorationStrategy(
                policy_type="Defect-Driven",
                r_md_mc=0,
                t_schedule=(300.0, tm * 0.5, 10000),
                n_defects=3,
                strain_range=0.02,
            )
        elif is_hard:
            return ExplorationStrategy(
                policy_type="Strain-Heavy",
                r_md_mc=0,
                t_schedule=(300.0, 500.0, 10000),
                n_defects=0,
                strain_range=0.15,
            )
        else:
            return ExplorationStrategy(
                policy_type="Standard-MD",
                r_md_mc=0,
                t_schedule=(300.0, 300.0, 10000),
                n_defects=0,
                strain_range=0.0,
            )
