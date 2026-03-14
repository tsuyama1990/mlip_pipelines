from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from src.domain_models.config import OracleConfig
from src.oracles.dft_oracle import DFTManager


def test_dft_manager_initialization() -> None:
    config = OracleConfig()
    manager = DFTManager(config)
    assert manager.config.kspacing == 0.05


def test_periodic_embedding() -> None:
    config = OracleConfig()
    manager = DFTManager(config)
    atoms = Atoms("Fe", positions=[(0, 0, 0)])

    embedded = manager._apply_periodic_embedding(atoms)

    # Must be Orthorhombic cell for embedding
    cell = embedded.get_cell()  # type: ignore[no-untyped-call]
    assert cell[0][1] == 0.0
    assert cell[0][2] == 0.0
    assert cell[1][0] == 0.0
    assert cell[1][2] == 0.0
    assert cell[2][0] == 0.0
    assert cell[2][1] == 0.0


def test_compute_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = OracleConfig()
    manager = DFTManager(config)

    from typing import Any, ClassVar

    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    class MockCalc(Calculator):  # type: ignore[misc]
        implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)  # type: ignore[no-untyped-call]
            self.parameters = {
                "input_data": {"electrons": {"mixing_beta": 0.7, "diagonalization": "david"}}
            }

        def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list[str] | None = None,
            system_changes: Any = None,
        ) -> None:
            if properties is None:
                properties = ["energy"]
            self.results = {
                "energy": -5.0,
                "forces": np.zeros((len(atoms) if atoms else 0, 3)),
                "stress": np.zeros(6),
            }

    atoms1 = Atoms("Fe", positions=[(0, 0, 0)])
    atoms2 = Atoms("Pt", positions=[(0, 0, 0)])

    # Monkeypatch the get_calculator logic to inject mock
    def mock_get_calculator(atoms: Atoms, work_dir: Path) -> MockCalc:
        return MockCalc()

    monkeypatch.setattr(manager, "_get_calculator", mock_get_calculator)

    results = manager.compute_batch([atoms1, atoms2], tmp_path)

    assert len(results) == 2
    assert results[0].calc is not None
    assert "energy" in results[0].calc.results
    assert results[0].calc.results["energy"] == -5.0


@pytest.fixture
def dft_oracle():
    from src.domain_models.config import OracleConfig
    from src.oracles.dft_oracle import DFTManager

    config = OracleConfig()
    return DFTManager(config)


def test_validate_pseudopotentials_absolute_path_error(dft_oracle, monkeypatch):
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", "relative/path")
    with pytest.raises(ValueError, match="Pseudopotential directory must be an absolute path"):
        dft_oracle._validate_pseudopotentials({"Fe"})


def test_validate_pseudopotentials_not_dir(dft_oracle, tmp_path, monkeypatch):
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("dummy")
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(not_a_dir))
    with pytest.raises(ValueError, match="Pseudopotential directory is not a valid directory"):
        dft_oracle._validate_pseudopotentials({"Fe"})


def test_validate_pseudopotentials_invalid_symbol(dft_oracle, tmp_path, monkeypatch):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(pseudo_dir))
    with pytest.raises(ValueError, match="Invalid element name"):
        dft_oracle._validate_pseudopotentials({"fe"})  # lowercase
    with pytest.raises(ValueError, match="Invalid chemical symbol detected"):
        dft_oracle._validate_pseudopotentials({"Xx"})  # not in atomic numbers


def test_validate_pseudopotentials_missing_file(dft_oracle, tmp_path, monkeypatch):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(pseudo_dir))
    with pytest.raises(FileNotFoundError, match="Pseudopotential file not found: Fe.upf"):
        dft_oracle._validate_pseudopotentials({"Fe"})


def test_validate_pseudopotentials_invalid_format(dft_oracle, tmp_path, monkeypatch):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    upf_file = pseudo_dir / "Fe.upf"
    upf_file.write_text("invalid content")
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(pseudo_dir))
    with pytest.raises(ValueError, match="Invalid UPF format for pseudopotential: Fe.upf"):
        dft_oracle._validate_pseudopotentials({"Fe"})


def test_validate_pseudopotentials_valid(dft_oracle, tmp_path, monkeypatch):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    upf_file = pseudo_dir / "Fe.upf"
    upf_file.write_text("<UPF valid format>")
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(pseudo_dir))
    pseudos = dft_oracle._validate_pseudopotentials({"Fe"})
    assert pseudos == {"Fe": "Fe.upf"}


def test_calculate_kpoints_invalid_cell(dft_oracle):
    atoms = Atoms("Fe", positions=[(0, 0, 0)], cell=[0, 0, 0])
    with pytest.raises(ValueError, match="Cell dimensions must be strictly positive and finite"):
        dft_oracle._calculate_kpoints(atoms)


def test_calculate_kpoints_too_many(dft_oracle):
    atoms = Atoms("Fe", positions=[(0, 0, 0)], cell=[1, 1, 1])
    # 0.001 kspacing will result in ~6283 points per dim -> way over 1000
    dft_oracle.config.kspacing = 0.001
    with pytest.raises(ValueError, match="exceeds maximum allowed points"):
        dft_oracle._calculate_kpoints(atoms)


def test_get_calculator(dft_oracle, tmp_path, monkeypatch):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    upf_file = pseudo_dir / "Fe.upf"
    upf_file.write_text("<UPF valid format>")
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(pseudo_dir))

    atoms = Atoms("Fe", positions=[(0, 0, 0)], cell=[20, 20, 20])
    # Mock Espresso import
    with patch("src.oracles.dft_oracle.Espresso") as mock_espresso:
        dft_oracle._get_calculator(atoms, tmp_path)
        assert mock_espresso.called


def test_compute_batch_self_healing_success(dft_oracle, tmp_path, monkeypatch):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    upf_file = pseudo_dir / "Fe.upf"
    upf_file.write_text("<UPF valid format>")
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(pseudo_dir))

    atoms = Atoms("Fe", positions=[(0, 0, 0)], cell=[20, 20, 20])

    class MockCalc:
        def __init__(self) -> None:
            self.parameters = {"input_data": {"electrons": {}}}
            self.call_count = 0

    calc = MockCalc()

    with patch.object(dft_oracle, "_get_calculator", return_value=calc):

        def mock_get_potential_energy(self):
            calc.call_count += 1
            if calc.call_count == 1:
                msg = "SCF Failed"
                raise Exception(msg)
            return 1.0

        with patch("ase.Atoms.get_potential_energy", mock_get_potential_energy):
            results = dft_oracle.compute_batch([atoms], tmp_path)
            assert len(results) == 1
            assert calc.parameters["input_data"]["electrons"]["mixing_beta"] == 0.3


def test_compute_batch_self_healing_retry_2_success(dft_oracle, tmp_path, monkeypatch):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    upf_file = pseudo_dir / "Fe.upf"
    upf_file.write_text("<UPF valid format>")
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(pseudo_dir))

    atoms = Atoms("Fe", positions=[(0, 0, 0)], cell=[20, 20, 20])

    class MockCalc:
        def __init__(self) -> None:
            self.parameters = {"input_data": {"electrons": {}}}
            self.call_count = 0

    calc = MockCalc()

    with patch.object(dft_oracle, "_get_calculator", return_value=calc):

        def mock_get_potential_energy(self):
            calc.call_count += 1
            if calc.call_count in (1, 2):
                msg = "SCF Failed"
                raise Exception(msg)
            return 1.0

        with patch("ase.Atoms.get_potential_energy", mock_get_potential_energy):
            results = dft_oracle.compute_batch([atoms], tmp_path)
            assert len(results) == 1
            assert calc.parameters["input_data"]["electrons"]["diagonalization"] == "cg"


def test_compute_batch_total_failure(dft_oracle, tmp_path, monkeypatch):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    upf_file = pseudo_dir / "Fe.upf"
    upf_file.write_text("<UPF valid format>")
    monkeypatch.setattr(dft_oracle.config, "pseudo_dir", str(pseudo_dir))

    atoms = Atoms("Fe", positions=[(0, 0, 0)], cell=[20, 20, 20])

    class MockCalc:
        def __init__(self) -> None:
            self.parameters = {"input_data": {"electrons": {}}}

    calc = MockCalc()

    with patch.object(dft_oracle, "_get_calculator", return_value=calc):
        with patch(
            "ase.Atoms.get_potential_energy", side_effect=Exception("SCF Failed completely")
        ):
            results = dft_oracle.compute_batch([atoms], tmp_path)
            assert len(results) == 0  # Struct failed completely
