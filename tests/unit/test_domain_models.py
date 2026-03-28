import pytest
from ase import Atoms

from src.domain_models.dtos import (
    CutoutResult,
    ExplorationStrategy,
    HaltInfo,
    MaterialFeatures,
    ValidationReport,
    ValidationScore,
)


def test_material_features_valid() -> None:
    feat = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    assert feat.elements == ["Fe", "Pt"]


# Note: Only a fragment for time limits... will just run pytest normally.


def test_exploration_strategy_valid() -> None:
    strategy = ExplorationStrategy(
        md_mc_ratio=0.5,
        t_max=500.0,
        n_defects=0.1,
        strain_range=0.15,
        policy_name="random"
    )
    assert strategy.policy_name == "random"
    assert strategy.t_max == 500.0


def test_halt_info_valid() -> None:
    # not halted
    info = HaltInfo(halted=False)
    assert not info.halted
    assert info.high_gamma_atoms is None

    # halted with atoms
    atoms = Atoms("Fe")
    info2 = HaltInfo(halted=True, high_gamma_atoms=[atoms])
    assert info2.halted
    assert info2.high_gamma_atoms is not None
    assert len(info2.high_gamma_atoms) == 1


def test_halt_info_invalid() -> None:
    # halted without high_gamma_atoms
    with pytest.raises(ValueError, match=".*high_gamma_atoms must not be None when halted is True.*"):
        _ = HaltInfo(halted=True, high_gamma_atoms=None)


def test_validation_report_valid() -> None:
    report = ValidationReport(
        passed=True,
        energy_rmse=0.01,
        force_rmse=0.02,
        stress_rmse=0.03,
        phonon_stable=True,
        mechanically_stable=True,
    )
    assert report.passed


def test_validation_report_basic_valid() -> None:
    report = ValidationReport(
        passed=False,
        reason="Test reason",
        energy_rmse=0.0,
        force_rmse=0.0,
        stress_rmse=0.0,
        phonon_stable=False,
        mechanically_stable=False,
    )
    assert not report.passed
    assert report.reason == "Test reason"


def test_validation_score_valid() -> None:
    score = ValidationScore(
        rmse_energy=0.01,
        rmse_forces=0.02,
        rmse_stress=0.03,
        born_stable=True,
        phonon_stable=True,
    )
    assert score.born_stable


def test_cutout_result_valid() -> None:
    atoms = Atoms("Fe")
    result = CutoutResult(cluster=atoms, passivation_atoms_added=2)
    assert result.passivation_atoms_added == 2
    assert result.cluster == atoms
