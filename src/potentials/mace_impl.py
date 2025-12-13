import torch
import numpy as np
from ase import Atoms
from mace.data import config_from_atoms
from mace.tools import torch_geometric, torch_tools
from typing import List, Optional, Tuple
from src.interfaces import AbstractPotential

class MacePotential(AbstractPotential):
    """
    MACE potential implementation with Last Layer Uncertainty (LLU).
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()  # Default to eval mode

        # Uncertainty components
        self.inv_covariance = None
        self.phi_hook = None
        self._hook_handle = None

        # We need to identify the readout layer to attach the hook.
        # Typically model.readouts is a ModuleList.
        # We want the last one, or we want the features *before* the final linear projection?
        # The spec says: "readout module (usually model.readouts[-1] or model.atomic_energies_fn is just before)"
        # "Capture node_feats (usually 128 or 256 dim)"
        # Looking at MACE code structure (standard):
        # Readout blocks usually take node features and output energy.
        # If we hook `model.readouts[-1]`, we need to inspect its forward.
        # Or we can hook the input to the last readout.

        # Let's try to find `readouts`.
        if hasattr(self.model, "readouts") and len(self.model.readouts) > 0:
             self.target_module = self.model.readouts[-1]
        else:
            # Fallback or error? For now assume standard MACE structure.
            # Some MACE versions structure differently.
            # If loaded from 'mace_mp', it might be wrapped.
            # Assuming standard structure for now.
            raise ValueError("Could not find readouts in MACE model.")

        self._register_hook()

    def _register_hook(self):
        """Register forward hook to capture last layer features."""
        def hook_fn(module, input, output):
            # Input to readout is typically (node_feats, ...)
            # We want node_feats.
            # Verify input structure.
            # MACE Readout forward signature: forward(self, x, ...) or similar.
            # input is a tuple. input[0] should be node features.
            if isinstance(input, tuple):
                self.phi_hook = input[0].detach()  # Capture features
            else:
                self.phi_hook = input.detach()

        if self._hook_handle:
            self._hook_handle.remove()

        self._hook_handle = self.target_module.register_forward_hook(hook_fn)

    def _atoms_to_batch(self, atoms_list: List[Atoms]):
        """Convert ASE Atoms to MACE batch."""
        configs = [config_from_atoms(a) for a in atoms_list]

        # Extract z_table and cutoff from model if available
        z_table = getattr(self.model, "z_table", None)
        if z_table is None:
            zs = getattr(self.model, "atomic_numbers", None)
            if zs is not None:
                from mace.tools.utils import AtomicNumberTable
                z_table = AtomicNumberTable([int(z) for z in zs])
            else:
                # If no z_table found, we might fail or infer.
                # For robustness, let's infer from data if possible, but MACE needs fixed table.
                # Assuming model has it for now as per MACE standard.
                raise ValueError("Model does not have z_table or atomic_numbers defined.")

        cutoff = getattr(self.model, "r_max", 5.0)

        from mace.data import AtomicData
        data_list = [AtomicData.from_config(c, z_table=z_table, cutoff=cutoff) for c in configs]

        batch = torch_geometric.batch.Batch.from_data_list(data_list)
        batch = batch.to(self.device)
        return batch.to_dict()

    def predict(self, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        batch_data = self._atoms_to_batch([atoms])

        # MACE forward returns a dictionary
        out = self.model(batch_data)

        energy = out["energy"].detach().cpu().numpy()[0] # Scalar? or shape (1,)
        forces = out["forces"].detach().cpu().numpy()    # (N, 3)
        stress = out["stress"].detach().cpu().numpy()[0] # (3, 3)

        return float(energy), forces, stress

    def train(self, training_data: list[Atoms]) -> None:
        """
        Fine-tune the model (Head Only) and update uncertainty covariance.
        """
        # 1. Head Only Training Setup
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze Readout parameters
        # We unfreeze the LAST readout.
        for param in self.target_module.parameters():
            param.requires_grad = True

        # Optimization loop setup
        optimizer = torch.optim.Adam(self.target_module.parameters(), lr=1e-3)

        self.model.train()

        # Simple training loop (e.g., 10 epochs for demonstration/simplicity,
        # or just one pass if "fine-tune" implies a single update?
        # The spec implies "Fine-tune ... with new data", likely a few steps.
        # But for 'train' method in Active Learning, usually we fit to the new data.
        # I will do a few epochs (e.g. 10).

        batch_data = self._atoms_to_batch(training_data)

        # Reference values (labels)
        # Assuming atoms in training_data have calculated energy/forces
        ref_energies = []
        ref_forces = []

        for a in training_data:
            # Check if calc exists and has results
            # The spec doesn't strictly say how labels are passed,
            # but usually they are in the atoms object or calculator.
            # We'll assume atoms.get_potential_energy() and atoms.get_forces() work.
            ref_energies.append(a.get_potential_energy())
            ref_forces.append(a.get_forces())

        ref_energy_tensor = torch.tensor(ref_energies, device=self.device, dtype=torch.float32)
        ref_forces_tensor = torch.tensor(np.concatenate(ref_forces), device=self.device, dtype=torch.float32)

        # Loss weights
        w_e = 1.0
        w_f = 10.0 # Common ratio

        # Epochs
        for _ in range(5):
            optimizer.zero_grad()
            out = self.model(batch_data)

            pred_e = out["energy"]
            pred_f = out["forces"]

            loss_e = torch.mean((pred_e - ref_energy_tensor)**2)
            loss_f = torch.mean((pred_f - ref_forces_tensor)**2)

            loss = w_e * loss_e + w_f * loss_f
            loss.backward()
            optimizer.step()

        # 2. Covariance Matrix Update (LLU)
        # Switch to eval to capture features without gradients
        self.model.eval()

        # Collect phi for all training atoms
        # We can reuse batch_data
        _ = self.model(batch_data) # Trigger hook
        # self.phi_hook contains features for all atoms in the batch (N_total, D)

        phi = self.phi_hook # (N_atoms_total, D)

        # C = sum phi * phi^T
        # Efficiently: C = phi.T @ phi
        C = torch.matmul(phi.T, phi)

        # Regularization
        lambda_reg = 1e-4
        C_reg = C + lambda_reg * torch.eye(C.shape[0], device=self.device)

        # Inverse
        # self.inv_covariance = torch.linalg.inv(C_reg)
        # Using pinv might be safer, but inv is faster if stable. Spec suggested inv or pinv.
        self.inv_covariance = torch.linalg.inv(C_reg)

    def get_uncertainty(self, atoms: Atoms) -> np.ndarray:
        if self.inv_covariance is None:
            # If not trained yet, return zeros or raise?
            # Return zeros or high uncertainty?
            # Let's return zeros with a warning or just zeros.
            return np.zeros(len(atoms))

        self.model.eval()
        batch_data = self._atoms_to_batch([atoms])

        _ = self.model(batch_data) # Trigger hook
        phi = self.phi_hook # (N_atoms, D)

        # u = phi^T * C^-1 * phi
        # We want diagonal of (phi @ C^-1 @ phi.T)
        # Efficient: sum((phi @ C^-1) * phi, dim=1)

        temp = torch.matmul(phi, self.inv_covariance) # (N, D)
        uncertainty = torch.sum(temp * phi, dim=1) # (N,)

        return uncertainty.detach().cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.model, path)

    def load(self, path: str) -> None:
        self.model = torch.load(path, map_location=self.device)
        self.model.eval()
        # Re-register hook
        if hasattr(self.model, "readouts") and len(self.model.readouts) > 0:
             self.target_module = self.model.readouts[-1]
        self._register_hook()
