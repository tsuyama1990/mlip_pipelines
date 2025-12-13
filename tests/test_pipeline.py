import pytest
import shutil
from pathlib import Path
from src.core.calculators.mace_factory import get_mace_calculator
from src.core.engines.relaxer import StructureRelaxer
from src.core.utils.io import save_results

@pytest.mark.slow
def test_pipeline_integration(tmp_path, default_settings, test_atoms):
    """
    Smoke test for the full pipeline (minus generation, using test_atoms).
    Uses 'small' MACE model on CPU for 2 steps.
    """
    # 1. Setup Settings
    default_settings.mace.model_path = "small" # Use small model
    default_settings.mace.device = "cpu"
    default_settings.relax.steps = 2
    default_settings.output_dir = tmp_path

    # 2. Load Calculator (Real download/load)
    # This might take time on first run
    try:
        calc = get_mace_calculator(default_settings.mace)
    except Exception as e:
        pytest.fail(f"Failed to load MACE model: {e}")

    test_atoms.calc = calc

    # 3. Run Relaxer
    relaxer = StructureRelaxer(default_settings)
    result = relaxer.run(test_atoms, run_id="smoke_test")

    # 4. Save Results
    output_dir = tmp_path / "smoke_test"
    save_results(result, default_settings, output_dir)

    # 5. Verify Outputs
    assert (output_dir / "final_structure.xyz").exists()
    assert (output_dir / "final_structure.cif").exists()
    assert (output_dir / "results.json").exists()
    assert (output_dir / "config.json").exists()

    # Verify JSON content
    import json
    with open(output_dir / "results.json") as f:
        data = json.load(f)
        assert data["steps"] <= 2
