import json
import pytest
from pathlib import Path
from core.energies import AtomicEnergyManager
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms

class MockE0Calculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, e0_map):
        super().__init__()
        self.e0_map = e0_map

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Simple logic: sum of per-atom energies
        syms = self.atoms.get_chemical_symbols()
        total_e = 0.0
        for s in syms:
            total_e += self.e0_map.get(s, 0.0)

        self.results['energy'] = total_e
        self.results['forces'] = [[0, 0, 0]] * len(self.atoms)

@pytest.fixture
def mock_calculator_factory():
    def factory(element):
        e0_map = {"Cu": -10.0, "H": -1.0}
        return MockE0Calculator(e0_map)
    return factory

@pytest.fixture
def setup_pseudo(tmp_path):
    # Setup dummy pseudo dir and SSSP
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()

    sssp_path = pseudo_dir / "sssp.json"
    sssp_data = {
        "Cu": {"filename": "Cu.upf"},
        "H": {"filename": "H.upf"}
    }
    with open(sssp_path, 'w') as f:
        json.dump(sssp_data, f)

    # Create dummy UPF files
    (pseudo_dir / "Cu.upf").touch()
    (pseudo_dir / "H.upf").touch()

    return pseudo_dir, sssp_path

def test_atomic_energy_manager(setup_pseudo, mock_calculator_factory):
    pseudo_dir, sssp_path = setup_pseudo

    manager = AtomicEnergyManager(str(pseudo_dir), str(sssp_path), mock_calculator_factory)

    # 1. First call: Should compute and cache
    energies = manager.get_atomic_energies(["Cu", "H"])
    assert energies["Cu"] == -10.0
    assert energies["H"] == -1.0

    # Verify cache file created
    cache_cu = pseudo_dir / "Cu.json"
    assert cache_cu.exists()

    with open(cache_cu, 'r') as f:
        data = json.load(f)
        assert data["energy"] == -10.0

    # 2. Second call: Should load from cache
    # Modify cache manually to verify it reads from cache
    with open(cache_cu, 'w') as f:
        json.dump({"element": "Cu", "energy": -99.9}, f)

    energies_2 = manager.get_atomic_energies(["Cu"])
    assert energies_2["Cu"] == -99.9 # Should reflect cache modification

def test_validate_missing_file(setup_pseudo, mock_calculator_factory):
    pseudo_dir, sssp_path = setup_pseudo
    # Delete a file
    (pseudo_dir / "Cu.upf").unlink()

    manager = AtomicEnergyManager(str(pseudo_dir), str(sssp_path), mock_calculator_factory)

    with pytest.raises(FileNotFoundError):
        manager.get_atomic_energies(["Cu"])
