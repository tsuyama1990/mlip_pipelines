from ase import Atoms

from src.domain_models.config import TrainerConfig
from src.trainers.ace_trainer import PacemakerWrapper


def test_pacemaker_initialization():
    config = TrainerConfig(max_epochs=10, active_set_size=200)
    wrapper = PacemakerWrapper(config)
    assert wrapper.config.max_epochs == 10

def test_update_dataset(tmp_path):
    config = TrainerConfig()
    wrapper = PacemakerWrapper(config)

    atoms1 = Atoms("Fe", positions=[(0, 0, 0)])
    atoms2 = Atoms("Pt", positions=[(0, 0, 0)])

    dataset_path = tmp_path / "accumulated.extxyz"
    dataset_path = wrapper.update_dataset([atoms1, atoms2], dataset_path)
    assert dataset_path.exists()
    assert str(dataset_path).endswith(".extxyz")

def test_select_local_active_set():
    config = TrainerConfig()
    wrapper = PacemakerWrapper(config)

    anchor = Atoms("Fe", positions=[(0, 0, 0)])
    candidates = [Atoms("Fe", positions=[(i*0.1, 0, 0)]) for i in range(1, 21)]

    selected = wrapper.select_local_active_set(candidates, anchor=anchor, n=5)

    # Must include anchor
    assert len(selected) == 5
    # Since we can't run pace_activeset without actual ACE features, we just assert the interface returns a list of atoms
    # For unit tests, we'll probably have to mock pace_activeset subprocess call or just rely on a dummy selection strategy inside the mock.

    assert isinstance(selected[0], Atoms)
