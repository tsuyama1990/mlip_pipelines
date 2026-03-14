with open("src/generators/adaptive_policy.py") as f:
    content = f.read()

content = content.replace("band_gap = 0.0", "band_gap = self.config.fallback_metal_band_gap")
content = content.replace(
    "melting_point = 1500.0  # Approx generic",
    "melting_point = self.config.fallback_metal_melting_point",
)
content = content.replace(
    "bulk_modulus = 150.0 # Approx generic",
    "bulk_modulus = self.config.fallback_metal_bulk_modulus",
)
content = content.replace("band_gap = 2.0", "band_gap = self.config.fallback_insulator_band_gap")
content = content.replace(
    "melting_point = 800.0", "melting_point = self.config.fallback_insulator_melting_point"
)
content = content.replace(
    "bulk_modulus = 50.0", "bulk_modulus = self.config.fallback_insulator_bulk_modulus"
)

content = content.replace(
    "features.initial_gamma_variance > 1.0",
    "features.initial_gamma_variance > self.config.uncertainty_variance_threshold",
)
content = content.replace(
    "features.band_gap <= 0.1", "features.band_gap <= self.config.metal_band_gap_threshold"
)
content = content.replace(
    "features.band_gap > 0.1", "features.band_gap > self.config.metal_band_gap_threshold"
)
content = content.replace(
    "features.bulk_modulus > 200.0",
    "features.bulk_modulus > self.config.hard_material_bulk_modulus_threshold",
)

content = content.replace(
    '''class FeatureExtractor:
    """Extracts material features using universal ML potentials (or rule-based fallbacks)."""

    def extract_features(self, elements: list[str]) -> MaterialFeatures:''',
    '''class FeatureExtractor:
    """Extracts material features using universal ML potentials (or rule-based fallbacks)."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        if config is None:
            self.config = PolicyConfig()
        else:
            self.config = config

    def extract_features(self, elements: list[str]) -> MaterialFeatures:''',
)

with open("src/generators/adaptive_policy.py", "w") as f:
    f.write(content)
