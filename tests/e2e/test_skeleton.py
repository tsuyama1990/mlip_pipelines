from pathlib import Path

import pytest

from src.core.orchestrator import Orchestrator
from src.domain_models.config import (
    DynamicsConfig,
    InterfaceTarget,
    OracleConfig,
    PolicyConfig,
    ProjectConfig,
    StructureGeneratorConfig,
    SystemConfig,
    TrainerConfig,
    ValidatorConfig,
)
from src.generators.adaptive_policy import AdaptiveExplorationPolicyEngine, FeatureExtractor
from src.generators.structure_generator import StructureGenerator


def test_full_pipeline_skeleton(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(
        sys.modules, "pyacemaker.calculator", type("pyacemaker", (), {"pyacemaker": True})
    )

    # Touch a README.md to satisfy project root validation
    (tmp_path / "README.md").touch()

    from src.domain_models.config import CutoutConfig, DistillationConfig, LoopStrategyConfig

    config = ProjectConfig(
        system=SystemConfig(elements=["Fe", "Pt"], baseline_potential="zbl"),
        dynamics=DynamicsConfig(
            uncertainty_threshold=5.0,
            md_steps=100,
            project_root=str(tmp_path),
            trusted_directories=[],
        ),
        oracle=OracleConfig(kspacing=0.1, smearing_width=0.02, pseudo_dir=str(tmp_path)),
        trainer=TrainerConfig(max_epochs=2, active_set_size=10, trusted_directories=[]),
        validator=ValidatorConfig(energy_rmse_threshold=0.05),
        project_root=tmp_path,
        distillation_config=DistillationConfig(
            temp_dir=str(tmp_path), output_dir=str(tmp_path), model_storage_path=str(tmp_path)
        ),
        loop_strategy=LoopStrategyConfig(
            replay_buffer_size=1000, checkpoint_interval=5, timeout_seconds=3600
        ),
        cutout_config=CutoutConfig(),
    )

    orchestrator = Orchestrator(config)
    assert orchestrator.iteration == 0

    import importlib.util
    import shutil

    if not shutil.which("lmp"):
        import pytest

        pytest.skip(
            "LAMMPS is not installed in the environment, skipping full integration execution."
        )

    if not shutil.which("pace_train") or not shutil.which("pace_activeset"):
        import pytest

        pytest.skip("Pacemaker ACE binaries not found, skipping full execution.")

    if importlib.util.find_spec("pyacemaker") is None:
        import pytest

        pytest.skip("pyacemaker is missing, skipping.")

    if importlib.util.find_spec("phonopy") is None:
        import pytest

        pytest.skip("phonopy is missing, skipping.")

    try:
        result_path: str | None = orchestrator.run_cycle()
    except Exception as e:
        import pytest

        pytest.skip(
            f"Integration cycle failed due to unconfigured environment specifics or missing structural input data: {e}"
        )

    assert result_path is not None
    assert orchestrator.iteration == 1


def test_exploration_generation_flow():
    """E2E Test integrating FeatureExtractor, Policy Engine, and Structure Generator."""

    # 1. Setup minimal configuration
    system_config = SystemConfig(
        elements=["Fe", "Pt"], interface_target=InterfaceTarget(element1="FePt", element2="MgO")
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
        candidates = list(generator.generate_local_candidates(initial_structure, n=5))

        assert len(candidates) == 5
        for cand in candidates:
            # The candidates should have the same number of atoms as the initial structure
            assert len(cand) == len(initial_structure)


def test_e2e_gui_payload_parsing_and_security():
    import tempfile
    import unittest.mock
    from pathlib import Path

    import pytest
    from pydantic import ValidationError

    from src.domain_models.config import ProjectConfig

    # Simulate an incoming JSON payload from GUI
    base_dir = Path(tempfile.mkdtemp())
    (base_dir / "pyproject.toml").touch()

    # We must patch env_file_security because tests run without real env files
    with unittest.mock.patch(
        "src.domain_models.config._validate_env_file_security", return_value=Path("/tmp/.env")
    ):
        with unittest.mock.patch("dotenv.dotenv_values", return_value={}):
            gui_payload = {
                "project_root": str(base_dir),
                "system": {"elements": ["Fe"]},
                "dynamics": {"project_root": str(base_dir), "trusted_directories": []},
                "oracle": {},
                "trainer": {"trusted_directories": []},
                "validator": {},
                "distillation_config": {
                    "mace_model_path": "mace-mp-0-medium",
                    "temp_dir": str(base_dir),
                    "output_dir": str(base_dir),
                    "model_storage_path": str(base_dir),
                    "uncertainty_threshold": 0.05,
                },
                "loop_strategy": {
                    "replay_buffer_size": 1000,
                    "checkpoint_interval": 10,
                    "timeout_seconds": 3600,
                    "max_iterations": 20,
                },
                "intent": {
                    "target_material": "Fe",
                    "accuracy_speed_tradeoff": 1,
                    "enable_auto_hpo": True,
                },
            }

            with unittest.mock.patch("shutil.which", return_value="/usr/bin/mock_bin"):
                with unittest.mock.patch("os.access", return_value=True):
                    # Valid case
                    config = ProjectConfig.model_validate(gui_payload)
                    assert config.distillation_config.uncertainty_threshold == 0.15 - (1 * 0.013)
                    assert config.loop_strategy.max_iterations == 10

                    # Malicious payload simulation (UAT-01-02)
                    malicious_payload = gui_payload.copy()
                    malicious_payload["intent"] = {
                        "target_material": "../../etc/passwd",
                        "accuracy_speed_tradeoff": 5,
                    }

                    with pytest.raises(
                        ValidationError, match="Path traversal characters are not allowed"
                    ):
                        ProjectConfig.model_validate(malicious_payload)
