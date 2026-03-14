import os
from pathlib import Path

from src.domain_models.dtos import ValidationReport
from src.validators.reporter import Reporter


def test_generate_html_report_passed(tmp_path: Path) -> None:
    report = ValidationReport(
        passed=True,
        reason=None,
        energy_rmse=0.001,
        force_rmse=0.02,
        stress_rmse=0.05,
        phonon_stable=True,
        mechanically_stable=True,
    )
    reporter = Reporter()
    save_path = tmp_path / "report.html"

    reporter.generate_html_report(report, save_path)

    assert save_path.exists()
    content = save_path.read_text()

    assert "Validation Report" in content
    assert '<span class="passed">PASS</span>' in content
    assert "0.00100 eV/atom" in content
    assert "0.02000 eV/A" in content
    assert "0.05000 GPa" in content


def test_generate_html_report_failed(tmp_path: Path) -> None:
    report = ValidationReport(
        passed=False,
        reason="Thresholds exceeded or instability detected.",
        energy_rmse=0.1,
        force_rmse=0.2,
        stress_rmse=0.5,
        phonon_stable=False,
        mechanically_stable=False,
    )
    reporter = Reporter()
    save_path = tmp_path / "report.html"

    reporter.generate_html_report(report, save_path)

    assert save_path.exists()
    content = save_path.read_text()

    assert "Validation Report" in content
    assert '<span class="failed">FAIL</span>' in content
    assert "Reason: Thresholds exceeded or instability detected." in content
    assert "0.10000 eV/atom" in content
    assert "0.20000 eV/A" in content
    assert "0.50000 GPa" in content
