import pytest
import torch
import torch.nn as nn
from mace.modules import MACE
from mace.data import AtomicData
from mace.tools.utils import AtomicNumberTable
import tempfile
import os
import numpy as np
from src.config.settings import Settings
from ase import Atoms

class MockMACE(nn.Module):
    """
    Minimal Mock MACE model that mimics the structure required by MacePotential.
    """
    def __init__(self):
        super().__init__()
        # Mock readouts: A list containing a module that processes features
        self.readouts = nn.ModuleList([
            nn.Linear(16, 1) # Simple linear layer as readout
        ])
        # Attributes expected by AtomicData conversion
        self.r_max = 5.0
        self.atomic_numbers = [1, 6, 8, 29] # H, C, O, Cu
        self.z_table = AtomicNumberTable(self.atomic_numbers)

    def forward(self, data):
        # Deterministic feature generation based on positions
        if 'positions' in data:
            pos = data['positions'] # (N, 3)
            # Simple projection to 16 dim to fake features
            # We use a fixed matrix to project 3 -> 16
            proj = torch.arange(3*16, dtype=torch.float32).view(3, 16).to(pos.device)
            # Normalize proj to avoid explosion
            proj = proj / 100.0

            node_feats = torch.matmul(pos, proj)
            # Normalize or sine to make it non-linear/bounded
            node_feats = torch.sin(node_feats)
        else:
            # Fallback
            num_nodes = data['node_attrs'].shape[0] if 'node_attrs' in data else 1
            node_feats = torch.zeros(num_nodes, 16)

        # Pass through readout
        # atomic_energies: (N, 1)
        atomic_energies = self.readouts[0](node_feats)

        # Construct output
        energy = torch.sum(atomic_energies).view(1) # Total energy
        num_nodes = node_feats.shape[0]
        forces = torch.zeros(num_nodes, 3).to(node_feats.device)
        stress = torch.zeros(1, 3, 3).to(node_feats.device)

        return {
            "energy": energy,
            "forces": forces,
            "stress": stress
        }

@pytest.fixture
def minimal_mace_model_path():
    """
    Creates a minimal MACE-like model and saves it to a temp file.
    Returns the path to the model.
    """
    model = MockMACE()

    # Save to temp file
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(model, path)

    yield path

    # Cleanup
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def default_settings():
    return Settings(
        mace={"model_path": "medium", "device": "cpu"},
        relax={"fmax": 0.05, "steps": 10},
        generator={"target_element": "Cu"}
    )

@pytest.fixture
def test_atoms():
    return Atoms('Cu4',
                 positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [0, 0, 1.8]],
                 cell=[4, 4, 4],
                 pbc=True)
