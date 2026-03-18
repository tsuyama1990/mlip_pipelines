import subprocess
import sys
from pathlib import Path

import pytest
from ase import Atoms

from src.domain_models.config import InterfaceTarget, PolicyConfig, StructureGeneratorConfig
from src.domain_models.dtos import MaterialFeatures
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine, FeatureExtractor
from src.generators.structure_generator import StructureGenerator


@pytest.mark.skip(
    reason="Headless execution of marimo notebooks can cause timeouts or missing dependency errors in CI"
)
def test_marimo_tutorial(tmp_path: Path) -> None:
    # Tests that the tutorial runs headlessly without errors
    tutorial_path = Path("tutorials/FePt_MgO_interface_energy.py")
    assert tutorial_path.exists()

    # Run the script directly since it's a valid python script via marimo
    res = subprocess.run(
        [sys.executable, "-m", "marimo", "run", str(tutorial_path)],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        check=False,
    )
    assert res.returncode == 0


def test_uat_02_01_adaptive_exploration_policy_evaluation():
    """UAT-02-01: Verify Adaptive Exploration Policy Engine."""
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
    """UAT-02-02: Verify Robust Interface Generation."""
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

    with pytest.raises(ValueError, match="Unsupported or disallowed interface element"):
        generator.generate_interface(target_invalid)


def test_uat_02_03_oom_protection():
    """UAT-02-03: Verify OOM Protection During Candidate Generation."""
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
    candidates = list(generator.generate_local_candidates(mod_atoms, n=200))

    # max(1, min(n, 100) // 10) = 10
    assert len(candidates) == 10


def test_uat_02_04_autonomous_feature_extraction():
    """UAT-02-04: Verify Autonomous Feature Extraction (Cold Start)."""
    extractor = FeatureExtractor()
    features = extractor.extract_features(["Fe", "Pt"])

    assert features.band_gap == 0.0
    assert features.melting_point > 0.0
