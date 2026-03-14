def append_safely(filepath, append_text):
    with open(filepath, "r") as f:
        content = f.read()
    if append_text.split("def test_")[1][:10] not in content:
        with open(filepath, "a") as f:
            f.write("\n" + append_text.strip() + "\n")


skel_text = """
from src.domain_models.config import InterfaceTarget, PolicyConfig, StructureGeneratorConfig
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine, FeatureExtractor
from src.generators.structure_generator import StructureGenerator

def test_exploration_generation_flow():
    \"\"\"E2E Test integrating FeatureExtractor, Policy Engine, and Structure Generator.\"\"\"

    # 1. Setup minimal configuration
    system_config = SystemConfig(
        elements=["Fe", "Pt"],
        interface_target=InterfaceTarget(element1="FePt", element2="MgO")
    )
    policy_config = PolicyConfig()
    struct_config = StructureGeneratorConfig()

    # 2. Extract features (Cold Start)
    extractor = FeatureExtractor()
    features = extractor.extract_features(system_config.elements)

    # Verify features are what we expect for FePt (metallic fallback)
    assert features.band_gap == 0.0
    assert features.melting_point > 0.0

    # 3. Decide Policy
    policy_engine = AdaptiveExplorationPolicyEngine(policy_config)
    strategy = policy_engine.decide_policy(features)

    # FePt is a multi-component metal -> High-MC Policy
    assert strategy.policy_name == "High-MC Policy"
    assert strategy.md_mc_ratio > 0

    # 4. Generate Structure (Initial Interface Generation)
    generator = StructureGenerator(struct_config)
    if system_config.interface_target:
        initial_structure = generator.generate_interface(system_config.interface_target)

        # Verify interface generated successfully
        symbols = initial_structure.get_chemical_symbols()
        assert "Fe" in symbols
        assert "Pt" in symbols
        assert "Mg" in symbols
        assert "O" in symbols
        assert len(initial_structure) > 4

        # 5. Simulate finding an uncertain structure and generating candidates
        candidates = generator.generate_local_candidates(initial_structure, n=5)

        assert len(candidates) == 5
        for cand in candidates:
            # The candidates should have the same number of atoms as the initial structure
            assert len(cand) == len(initial_structure)
"""

uat_text = """
from ase import Atoms
from src.domain_models.config import InterfaceTarget, PolicyConfig, StructureGeneratorConfig
from src.domain_models.dtos import MaterialFeatures
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine, FeatureExtractor
from src.generators.structure_generator import StructureGenerator

def test_uat_02_01_adaptive_exploration_policy_evaluation():
    \"\"\"UAT-02-01: Verify Adaptive Exploration Policy Engine.\"\"\"
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)

    # Metal Scenario
    features_metal = MaterialFeatures(elements=["Fe", "Pt"], melting_point=1000.0, band_gap=0.0)
    strategy_metal = engine.decide_policy(features_metal)
    assert strategy_metal.md_mc_ratio > 0
    assert strategy_metal.policy_name == "High-MC Policy"

    # Insulator Scenario
    features_insulator = MaterialFeatures(elements=["Mg", "O"], melting_point=1000.0, band_gap=2.0)
    strategy_insulator = engine.decide_policy(features_insulator)
    assert strategy_insulator.n_defects > 0
    assert strategy_insulator.policy_name == "Defect-Driven Policy"

def test_uat_02_02_robust_interface_generation():
    \"\"\"UAT-02-02: Verify Robust Interface Generation.\"\"\"
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    # Valid FePt/MgO
    target_valid = InterfaceTarget(element1="FePt", element2="MgO")
    interface_atoms = generator.generate_interface(target_valid)

    symbols = interface_atoms.get_chemical_symbols()
    assert "Fe" in symbols
    assert "Pt" in symbols
    assert "Mg" in symbols
    assert "O" in symbols

    # Invalid Unobtainium
    target_invalid = InterfaceTarget(element1="Unobtainium", element2="Fe")
    import pytest
    with pytest.raises(ValueError, match="Invalid or unsupported element target"):
        generator.generate_interface(target_invalid)

def test_uat_02_03_oom_protection():
    \"\"\"UAT-02-03: Verify OOM Protection During Candidate Generation.\"\"\"
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    # Massive object
    massive_atoms = Atoms("Fe" * 50000)
    import pytest
    with pytest.raises(ValueError, match="too large for rattling"):
        generator.generate_local_candidates(massive_atoms)

    # Moderately large object scaling
    mod_atoms = Atoms("Fe" * 2000, positions=[(0, 0, 0)] * 2000)
    mod_atoms.set_cell([10, 10, 10])
    candidates = generator.generate_local_candidates(mod_atoms, n=200)

    # max(1, min(n, 100) // 10) = 10
    assert len(candidates) == 10

def test_uat_02_04_autonomous_feature_extraction():
    \"\"\"UAT-02-04: Verify Autonomous Feature Extraction (Cold Start).\"\"\"
    extractor = FeatureExtractor()
    features = extractor.extract_features(["Fe", "Pt"])

    assert features.band_gap == 0.0
    assert features.melting_point > 0.0
"""

adapt_text = """
from src.generators.adaptive_policy import FeatureExtractor
import pytest

def test_feature_extractor_fallback():
    \"\"\"Test that FeatureExtractor correctly falls back to mock logic.\"\"\"
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
    \"\"\"Test invalid elements.\"\"\"
    extractor = FeatureExtractor()
    with pytest.raises(ValueError, match="cannot be empty"):
        extractor.extract_features([])

    with pytest.raises(ValueError, match="Invalid element"):
        extractor.extract_features(["Unobtainium"])

def test_adaptive_policy_engine_strain():
    \"\"\"Rule 4: Hard material -> Strain-Heavy Policy\"\"\"
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)
    # Band gap 0.0, single component (bypasses rule 2), high bulk modulus
    features = MaterialFeatures(elements=["C"], melting_point=4000.0, band_gap=0.0, bulk_modulus=500.0)

    strategy = engine.decide_policy(features)
    assert strategy.policy_name == "Strain-Heavy Policy"
    assert strategy.strain_range == config.strain_heavy_range

def test_adaptive_policy_engine_default():
    \"\"\"Default Strategy\"\"\"
    config = PolicyConfig()
    engine = AdaptiveExplorationPolicyEngine(config)
    # Band gap 0.0, single component, normal bulk modulus
    features = MaterialFeatures(elements=["Fe"], melting_point=1000.0, band_gap=0.0, bulk_modulus=100.0)

    strategy = engine.decide_policy(features)
    assert strategy.policy_name == "Standard"
    assert strategy.t_max == config.default_t_max_scale * 1000.0
"""

struct_text = """
from ase.build import bulk

def test_generate_local_candidates_oom_protection():
    \"\"\"Test OOM protection blocks large structures.\"\"\"
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    # Too large
    huge_atoms = Atoms("Fe" * 10001)
    import pytest
    with pytest.raises(ValueError, match="too large for rattling"):
        generator.generate_local_candidates(huge_atoms)

def test_generate_local_candidates_scaling():
    \"\"\"Test candidate count scaling for moderately large structures.\"\"\"
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    # 2000 atoms, requests 50 candidates
    # Actual n should be max(1, 50 // 10) = 5
    mod_atoms = Atoms("Fe" * 2000, positions=[(0, 0, 0)] * 2000)
    mod_atoms.set_cell([10, 10, 10])

    candidates = generator.generate_local_candidates(mod_atoms, n=50)
    assert len(candidates) == 5

def test_generate_local_candidates_normal():
    \"\"\"Test normal candidate generation.\"\"\"
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    atoms = Atoms("Fe", positions=[(0, 0, 0)])
    atoms.set_cell([10, 10, 10])

    candidates = generator.generate_local_candidates(atoms, n=5)
    assert len(candidates) == 5

    # Rattling should move the atom
    for cand in candidates:
        assert cand.positions[0][0] != 0.0 or cand.positions[0][1] != 0.0

def test_generate_interface_invalid():
    \"\"\"Test interface generation with invalid elements.\"\"\"
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    target = InterfaceTarget(element1="Unobtainium", element2="Fe")
    import pytest
    with pytest.raises(ValueError, match="Invalid or unsupported element target"):
        generator.generate_interface(target)

def test_generate_interface_valid():
    \"\"\"Test interface generation with valid elements.\"\"\"
    config = StructureGeneratorConfig()
    generator = StructureGenerator(config)

    target = InterfaceTarget(element1="FePt", element2="MgO")

    interface_atoms = generator.generate_interface(target)

    symbols = interface_atoms.get_chemical_symbols()
    assert "Fe" in symbols
    assert "Pt" in symbols
    assert "Mg" in symbols
    assert "O" in symbols

    # Check that it's a stacked structure (more than just the bulk atoms)
    assert len(interface_atoms) > 4
"""

append_safely("tests/e2e/test_skeleton.py", skel_text)
append_safely("tests/uat/test_tutorial.py", uat_text)
append_safely("tests/unit/test_adaptive_policy.py", adapt_text)
append_safely("tests/unit/test_structure_generator.py", struct_text)
