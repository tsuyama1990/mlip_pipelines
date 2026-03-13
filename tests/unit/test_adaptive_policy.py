import pytest
from src.generators.adaptive_policy import AdaptivePolicy

def test_adaptive_policy_high_mc() -> None:
    dna = {"elements": ["Fe", "Pt"]}
    props = {"band_gap": 0.0, "melting_point": 1000.0, "bulk_modulus": 100.0}
    policy = AdaptivePolicy(dna, props)
    strategy = policy.generate_strategy()

    assert strategy.policy_type == "High-MC"
    assert strategy.r_md_mc == 100
    assert strategy.t_schedule == (300.0, 800.0, 20000)
    assert strategy.n_defects == 1
    assert strategy.strain_range == 0.05

def test_adaptive_policy_defect_driven() -> None:
    dna = {"elements": ["Fe"]}
    props = {"band_gap": 1.0, "melting_point": 1000.0, "bulk_modulus": 100.0}
    policy = AdaptivePolicy(dna, props)
    strategy = policy.generate_strategy()

    assert strategy.policy_type == "Defect-Driven"
    assert strategy.r_md_mc == 0
    assert strategy.t_schedule == (300.0, 500.0, 10000)
    assert strategy.n_defects == 3
    assert strategy.strain_range == 0.02

def test_adaptive_policy_strain_heavy() -> None:
    dna = {"elements": ["Fe"]}
    props = {"band_gap": 0.0, "melting_point": 1000.0, "bulk_modulus": 200.0}
    policy = AdaptivePolicy(dna, props)
    strategy = policy.generate_strategy()

    assert strategy.policy_type == "Strain-Heavy"
    assert strategy.r_md_mc == 0
    assert strategy.t_schedule == (300.0, 500.0, 10000)
    assert strategy.n_defects == 0
    assert strategy.strain_range == 0.15

def test_adaptive_policy_standard_md() -> None:
    dna = {"elements": ["Fe"]}
    props = {"band_gap": 0.0, "melting_point": 1000.0, "bulk_modulus": 100.0}
    policy = AdaptivePolicy(dna, props)
    strategy = policy.generate_strategy()

    assert strategy.policy_type == "Standard-MD"
    assert strategy.r_md_mc == 0
    assert strategy.t_schedule == (300.0, 300.0, 10000)
    assert strategy.n_defects == 0
    assert strategy.strain_range == 0.0
