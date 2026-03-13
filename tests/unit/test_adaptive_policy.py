from src.domain_models.config import PolicyConfig
from src.domain_models.dtos import MaterialFeatures
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine


def test_adaptive_policy_metal():
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


def test_adaptive_policy_insulator():
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


def test_adaptive_policy_high_uncertainty():
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
