import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from ase import Atoms
from omegaconf import OmegaConf

# --- Mock MACE Model ---
class MockBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        return torch.matmul(x, self.weight)

class MockReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 1))

    def forward(self, x):
        return torch.matmul(x, self.weight)

class MockZTable:
    def __init__(self, zs):
        self.zs = zs
        self.z_map = {z: i for i, z in enumerate(zs)}

    def z_to_index(self, z):
        return self.z_map.get(int(z), 0)

    def __len__(self):
        return len(self.zs)

class MockMACE(torch.nn.Module):
    """
    A deterministic Mock MACE model for testing.
    """
    def __init__(self):
        super().__init__()
        self.readouts = torch.nn.ModuleList([MockReadout()])
        self.backbone = MockBackbone()
        self.r_max = 5.0
        self.atomic_numbers = [1, 6, 8, 26] # H, C, O, Fe
        self.z_table = MockZTable(self.atomic_numbers)
        
    def forward(self, data, training=False, compute_force=True, **kwargs):
        # Deterministic dummy output
        positions = data.get('positions')
        if positions is None:
             node_feats = data.get('node_attrs')
             N = node_feats.shape[0] if node_feats is not None else 2
             emb = torch.randn(N, 16)
        else:
            # Deterministic embedding based on position
            # (N, 3) 
            # Pad positions to 16
            x = torch.nn.functional.pad(positions, (0, 13)) 
            emb = self.backbone(x)
        
        # Pass through readout
        out = self.readouts[0](emb) # (N, 1) E_i per atom
        
        # Energy should be (Batch,) -> (1,) for single structure batch
        # Assuming single batch for tests here
        E = out.sum().view(-1)
        
        return {
            "energy": E, 
            "forces": torch.zeros_like(positions) if positions is not None else None, # Dummy forces
            "node_features": emb # Needed for LLU
        }
        
    def to(self, device):
        return self
        
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)


# --- Fixtures ---

@pytest.fixture
def test_atoms():
    """Simple Atoms object for testing."""
    return Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10], pbc=True)

@pytest.fixture
def default_settings():
    """
    Default configuration structure using OmegaConf.
    """
    conf = OmegaConf.create({
        "generator": {
            "config": {
                "type": "alloy"
            },
            "target_element": "Si",
            "supercell_size": 1,
            "vacancy_concentration": 0.0,
            "elements": [] 
        },
        "mace": {
            "model_path": "medium",
            "device": "cpu",
            "default_dtype": "float64"
        },
        "relax": {
            "steps": 5,
            "fmax": 0.01,
            "optimizer": "LBFGS" # Added missing key
        },
        "output_dir": "test_output"
    })
    return conf

@pytest.fixture
def minimal_mace_model_path(tmp_path, mocker):
    """
    Creates a dummy model file and mocks torch.load to return a MockMACE model.
    """
    model_path = tmp_path / "mock_mace.model"
    model_path.touch()
    
    mock_model_instance = MockMACE()
    
    original_load = torch.load
    
    def side_effect(f, *args, **kwargs):
        if str(f) == str(model_path):
            return mock_model_instance
        return original_load(f, *args, **kwargs)
        
    mocker.patch("torch.load", side_effect=side_effect)
    
    return str(model_path)

@pytest.fixture
def mock_potential(minimal_mace_model_path):
    """Fixture to provide a MacePotential initialized with the mock."""
    from potentials.mace_impl import MacePotential
    return MacePotential(minimal_mace_model_path)
