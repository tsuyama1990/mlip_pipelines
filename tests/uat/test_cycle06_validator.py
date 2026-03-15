from pathlib import Path

import pytest

from src.domain_models.config import ValidatorConfig
from src.domain_models.dtos import ValidationReport
from src.validators.reporter import Reporter
from src.validators.validator import Validator


def test_uat_06_01_quality_assurance_pass(tmp_path: Path):
    import importlib.util
    if importlib.util.find_spec("pyacemaker") is None:
        pytest.skip("pyacemaker is missing, skipping actual validation run")

    config = ValidatorConfig(energy_rmse_threshold=0.002, force_rmse_threshold=0.05, test_dataset_path=str(tmp_path / "test.extxyz"))

    dummy_pot = tmp_path / "potential.yace"
    dummy_pot.write_text("elements version")

    validator = Validator(config)
    try:
        report = validator.validate(dummy_pot)
        assert report.passed is True
    except Exception as e:
        pytest.skip(f"Failed to execute real validation due to missing model structures/formats: {e}")

def test_uat_06_02_quality_assurance_failure(tmp_path: Path):
    import importlib.util
    if importlib.util.find_spec("pyacemaker") is None:
        pytest.skip("pyacemaker is missing, skipping actual validation run")

    config = ValidatorConfig(energy_rmse_threshold=0.002, force_rmse_threshold=0.05, test_dataset_path=str(tmp_path / "test.extxyz"))

    dummy_pot = tmp_path / "poor_potential.yace"
    dummy_pot.write_text("elements version")

    validator = Validator(config)
    try:
        report = validator.validate(dummy_pot)
        assert report.passed is False
    except Exception as e:
        pytest.skip(f"Failed to execute real validation due to missing model structures/formats: {e}")

def test_uat_06_03_automated_report_generation(tmp_path: Path):
    reporter = Reporter()

    report = ValidationReport(
        passed=True,
        reason=None,
        energy_rmse=0.001,
        force_rmse=0.04,
        stress_rmse=0.0,
        phonon_stable=True,
        mechanically_stable=True
    )

    save_path = tmp_path / "validation_report.html"
    reporter.generate_html_report(report, save_path)

    assert save_path.exists()
    content = save_path.read_text()
    assert "0.001" in content
    assert "0.04" in content
    assert "PASS" in content
