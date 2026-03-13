import pytest
from ase import Atoms
from pydantic import ValidationError

from src.domain_models.dtos import ExplorationStrategy, HaltInfo, MaterialFeatures, ValidationReport


def test_material_features_valid():
    feat = MaterialFeatures(
        elements=["Fe", "Pt"], band_gap=0.0, bulk_modulus=250.0, melting_point=1600.0
    )
    assert feat.elements == ["Fe", "Pt"]


def test_material_features_invalid():
    with pytest.raises(ValidationError):
        MaterialFeatures(elements=["Si"], band_gap=-1.0)  # ge=0.0

    with pytest.raises(ValidationError):
        MaterialFeatures(elements=["Fe"], extra_field="bad")  # extra="forbid"


def test_exploration_strategy():
    strategy = ExplorationStrategy(policy_name="Defect-Driven Policy", n_defects=0.05)
    assert strategy.policy_name == "Defect-Driven Policy"
    assert strategy.n_defects == 0.05


def test_halt_info_with_atoms():
    atoms1 = Atoms("Fe", positions=[(0, 0, 0)])
    atoms2 = Atoms("Pt", positions=[(0, 0, 0)])

    halt = HaltInfo(halted=True, high_gamma_atoms=[atoms1, atoms2], max_gamma=6.5)
    assert halt.halted is True
    assert halt.max_gamma == 6.5
    assert len(halt.high_gamma_atoms) == 2


def test_validation_report_invalid():
    with pytest.raises(ValidationError):
        ValidationReport(
            passed=True,
            energy_rmse=0.005,
            force_rmse=0.03,
            stress_rmse=0.1,
            phonon_stable=True,
            mechanically_stable=True,
            extra_field="invalid",  # extra="forbid"
        )
