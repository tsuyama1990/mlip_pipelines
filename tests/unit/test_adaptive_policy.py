import pytest

from src.domain_models.config import PolicyConfig
from src.domain_models.dtos import MaterialFeatures
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine, FeatureExtractor


def test_adaptive_policy_metal() -> None:
    # FePt-like material
    features = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)
    strategy = engine.decide_policy(features)

    # Metal & Multi-component should result in High-MC Policy
    assert strategy.policy_name == "High-MC Policy"
    assert strategy.md_mc_ratio > 0.0
    assert strategy.t_max >= 0.8 * features.melting_point


def test_adaptive_policy_insulator() -> None:
    # MgO-like material
    features = MaterialFeatures(
        elements=["Mg", "O"], band_gap=7.8, bulk_modulus=160.0, melting_point=3125.0
    )
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)
    strategy = engine.decide_policy(features)

    # Insulator should trigger Defect-Driven Policy
    assert strategy.policy_name == "Defect-Driven Policy"
    assert strategy.n_defects > 0.0
    assert strategy.md_mc_ratio == 0.0


def test_adaptive_policy_high_uncertainty() -> None:
    features = MaterialFeatures(
        elements=["Ti"],
        band_gap=0.0,
        bulk_modulus=110.0,
        melting_point=1941.0,
        initial_gamma_variance=10.0,
    )
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)
    strategy = engine.decide_policy(features)

    # High initial uncertainty
    assert strategy.policy_name == "Cautious Exploration"
    assert strategy.t_max < features.melting_point


def test_feature_extractor_fallback():
    """Test that FeatureExtractor correctly falls back to mock logic."""
    extractor = FeatureExtractor()
    features = extractor.extract_features(["Fe"])

    # Since we don't have matgl/chgnet installed in test env (or they'll fail in UAT),
    # this should hit the fallback logic.
    assert features.elements == ["Fe"]
    assert features.band_gap == 0.0
    assert features.melting_point == 1500.0

    # Insulator fallback
    features = extractor.extract_features(["Mg", "O"])
    assert features.elements == ["Mg", "O"]
    assert features.band_gap == 2.0
    assert features.melting_point == 800.0


def test_feature_extractor_invalid():
    """Test invalid elements."""
    extractor = FeatureExtractor()
    with pytest.raises(ValueError, match="cannot be empty"):
        extractor.extract_features([])

    with pytest.raises(ValueError, match="Invalid element"):
        extractor.extract_features(["Unobtainium"])


def test_adaptive_policy_engine_strain():
    """Rule 4: Hard material -> Strain-Heavy Policy"""
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)
    # Band gap 0.0, single component (bypasses rule 2), high bulk modulus
    features = MaterialFeatures(
        elements=["C"], melting_point=4000.0, band_gap=0.0, bulk_modulus=500.0
    )

    strategy = engine.decide_policy(features)
    assert strategy.policy_name == "Strain-Heavy Policy"
    assert strategy.strain_range == config.strain_heavy_range


def test_adaptive_policy_engine_default():
    """Default Strategy"""
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)
    # Band gap 0.0, single component, normal bulk modulus
    features = MaterialFeatures(
        elements=["Fe"], melting_point=1000.0, band_gap=0.0, bulk_modulus=100.0
    )

    strategy = engine.decide_policy(features)
    assert strategy.policy_name == "Standard"
    assert strategy.t_max == config.default_t_max_scale * 1000.0
