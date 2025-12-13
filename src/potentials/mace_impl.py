
import numpy as np
import torch
from ase import Atoms
from mace.data import config_from_atoms
from mace.tools import torch_geometric

from src.core.interfaces import AbstractPotential


class MacePotential(AbstractPotential):
    """
    MACE potential implementation with Last Layer Uncertainty (LLU).
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.foundation_model_path = None
        self.u_max = 1.0
        self.inv_covariance = None
        self.model = None
        self._hook_handle = None
        self.phi_hook = None

        # Load logic handled here to support init from path
        self.load(model_path)

    def _register_hook(self) -> None:
        """Register forward hook to capture last layer features."""
        def hook_fn(module, input, output):
            # Input to readout is typically (node_feats, ...)
            if isinstance(input, tuple):
                self.phi_hook = input[0].detach()  # Capture features
            else:
                self.phi_hook = input.detach()

        if self._hook_handle:
            self._hook_handle.remove()

        self._hook_handle = self.target_module.register_forward_hook(hook_fn)

    def _atoms_to_batch(self, atoms_list: list[Atoms]) -> dict:
        """
        Convert ASE Atoms to MACE batch.

        Parameters
        ----------
        atoms_list : List[Atoms]
            List of ASE atoms objects.

        Returns
        -------
        Dict
            MACE batch dictionary.
        """
        configs = [config_from_atoms(a) for a in atoms_list]

        # Extract z_table and cutoff from model if available
        z_table = getattr(self.model, "z_table", None)
        if z_table is None:
            zs = getattr(self.model, "atomic_numbers", None)
            if zs is not None:
                from mace.tools.utils import AtomicNumberTable
                z_table = AtomicNumberTable([int(z) for z in zs])
            else:
                raise ValueError("Model does not have z_table or atomic_numbers defined.")

        cutoff = getattr(self.model, "r_max", 5.0)
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()

        from mace.data import AtomicData
        data_list = [AtomicData.from_config(c, z_table=z_table, cutoff=cutoff) for c in configs]

        batch = torch_geometric.batch.Batch.from_data_list(data_list)
        batch = batch.to(self.device)
        return batch.to_dict()

    def predict(self, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Return (energy [eV], forces [eV/A], stress [eV/A^3]).
        """
        self.model.eval()
        batch_data = self._atoms_to_batch([atoms])

        # MACE forward returns a dictionary
        out = self.model(batch_data)

        energy = out["energy"].detach().cpu().numpy()[0] # Scalar? or shape (1,)
        forces = out["forces"].detach().cpu().numpy()    # (N, 3)

        stress = out.get("stress")
        if stress is not None:
             stress = stress.detach().cpu().numpy()[0] # (3, 3)
        else:
             stress = np.zeros((3, 3))

        return float(energy), forces, stress

    def train(self, training_data: list[Atoms], atomic_energies: dict[str, float] | None = None, energy_weight: float = 1.0, forces_weight: float = 10.0, **kwargs) -> None:
        """
        Fine-tune the model (Head Only) and update uncertainty covariance.

        Parameters
        ----------
        training_data : list[Atoms]
            New structures for training.
        atomic_energies : Optional[Dict[str, float]]
            Dictionary of isolated atomic energies (E0) for referencing.
        energy_weight : float
            Weight for energy loss term.
        forces_weight : float
            Weight for forces loss term.
        """
        self._train_head(training_data, energy_weight=energy_weight, forces_weight=forces_weight)
        self._update_uncertainty_covariance(training_data)

    def _train_head(self, training_data: list[Atoms], energy_weight: float = 1.0, forces_weight: float = 10.0) -> None:
        """
        Perform Head-Only training.
        """
        # Remove hook during training to prevent graph retention issues
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

        try:
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze Readout parameters
            for param in self.target_module.parameters():
                param.requires_grad = True

            # Optimization loop setup
            optimizer = torch.optim.Adam(self.target_module.parameters(), lr=1e-3)

            self.model.train()

            ref_energies = []
            ref_forces = []

            for a in training_data:
                ref_energies.append(a.get_potential_energy())
                ref_forces.append(a.get_forces())

            ref_energy_tensor = torch.tensor(ref_energies, device=self.device, dtype=torch.float32)
            ref_forces_tensor = torch.tensor(np.concatenate(ref_forces), device=self.device, dtype=torch.float32)

            # Epochs
            for _ in range(5):
                batch_data = self._atoms_to_batch(training_data)
                optimizer.zero_grad()

                # Pass 1: Energy only (avoids graph retention issues from forces calc)
                out_e = self.model(batch_data, compute_force=False)
                pred_e = out_e["energy"]
                loss_e = torch.mean((pred_e - ref_energy_tensor)**2)

                # Pass 2: Forces (creates separate graph branch)
                # Note: We must recreate batch_data because MACE modifies it in-place
                batch_data_f = self._atoms_to_batch(training_data)
                out_f = self.model(batch_data_f, compute_force=True)
                pred_f = out_f["forces"]
                loss_f = torch.mean((pred_f - ref_forces_tensor)**2)

                loss = energy_weight * loss_e + forces_weight * loss_f
                loss.backward()
                optimizer.step()
        finally:
            # Re-register hook
            self._register_hook()

    def _update_uncertainty_covariance(self, training_data: list[Atoms]) -> None:
        """
        Update the covariance matrix for LLU.
        """
        self.model.eval()
        batch_data = self._atoms_to_batch(training_data)

        _ = self.model(batch_data) # Trigger hook
        phi = self.phi_hook # (N_atoms_total, D)

        cov = torch.matmul(phi.T, phi)

        # Regularization
        lambda_reg = 1e-4
        cov_reg = cov + lambda_reg * torch.eye(cov.shape[0], device=self.device)

        # Robust Inverse
        self.inv_covariance = torch.linalg.pinv(cov_reg)

        # Calculate u_max
        temp = torch.matmul(phi, self.inv_covariance) # (N, D)
        scores = torch.sum(temp * phi, dim=1) # (N,)
        self.u_max = torch.max(scores).item()

    def get_uncertainty(self, atoms: Atoms) -> np.ndarray:
        """
        Return per-atom uncertainty scores.
        """
        if self.inv_covariance is None:
            return np.zeros(len(atoms))

        self.model.eval()
        batch_data = self._atoms_to_batch([atoms])

        _ = self.model(batch_data) # Trigger hook
        phi = self.phi_hook # (N_atoms, D)

        temp = torch.matmul(phi, self.inv_covariance) # (N, D)
        uncertainty = torch.sum(temp * phi, dim=1) # (N,)

        # Normalize
        u_max = self.u_max if (self.u_max is not None and self.u_max > 0) else 1.0
        normalized_uncertainty = uncertainty / u_max

        return normalized_uncertainty.detach().cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save the model and uncertainty state.
        We save a dictionary wrapper to ensure persistence of inv_covariance.
        """
        # Remove hook before saving to avoid pickling errors with local functions
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

        state = {
            "foundation_model_path": self.foundation_model_path,
            "model_state": self.model.state_dict(),
            "inv_covariance": self.inv_covariance,
            "u_max": self.u_max
        }
        torch.save(state, path)

        # Restore hook after save so this instance remains usable
        self._register_hook()

    def load(self, path: str) -> None:
        """
        Load the model and uncertainty state.
        Handles both raw model files (legacy) and our wrapper dict.
        """
        loaded = torch.load(path, map_location=self.device)

        if isinstance(loaded, dict) and "model_state" in loaded:
            # New format
            foundation_path = loaded.get("foundation_model_path", None)
            if not foundation_path:
                raise ValueError("Invalid checkpoint: missing foundation_model_path")

            # Reconstruct architecture from foundation path
            self.model = torch.load(foundation_path, map_location=self.device)
            self.model.load_state_dict(loaded["model_state"])

            self.inv_covariance = loaded.get("inv_covariance", None)
            self.u_max = loaded.get("u_max", 1.0)
            self.foundation_model_path = foundation_path

        elif isinstance(loaded, dict) and "model" in loaded:
             # Intermediate format (from previous incorrect implementation attempt) or unexpected dict
             # Assuming "model" key holds the model object as per my previous reading of the file
             self.model = loaded["model"]
             self.inv_covariance = loaded.get("inv_covariance", None)
             self.u_max = 1.0
             self.foundation_model_path = path # This is technically wrong if path is a checkpoint, but legacy handling is fuzzy.

        elif isinstance(loaded, torch.nn.Module):
            # Raw model (Foundation model)
            self.model = loaded
            self.inv_covariance = None
            self.u_max = 1.0
            self.foundation_model_path = path
        else:
            raise ValueError(f"Unknown model format at {path}")

        self.model.to(self.device)
        self.model.eval()

        self.phi_hook = None
        self._hook_handle = None

        # Re-register hook
        if hasattr(self.model, "readouts") and len(self.model.readouts) > 0:
             self.target_module = self.model.readouts[-1]
        else:
             raise ValueError("Could not find readouts in MACE model.")

        self._register_hook()
