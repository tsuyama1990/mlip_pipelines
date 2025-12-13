import sys
import pytest
from pathlib import Path
from ase import Atoms
from ase.build import bulk
from src.config.settings import Settings, GeneratorSettings, MACESettings, RelaxationSettings

# Ensure source and external modules are in path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
EXTERNAL_DIR = SRC_DIR / "external" / "mlip_struc_gen" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))

@pytest.fixture
def test_atoms():
    """Returns a simple Silicon crystal."""
    return bulk("Si", "diamond", a=5.43)

@pytest.fixture
def default_settings():
    """Returns a default Settings object."""
    return Settings(
        mace=MACESettings(model_path="medium", device="cpu", default_dtype="float64"),
        relax=RelaxationSettings(fmax=0.01, steps=10, optimizer="LBFGS"),
        generator=GeneratorSettings(target_element="Si", supercell_size=1, vacancy_concentration=0.0),
        output_dir=Path("tests/output")
    )
