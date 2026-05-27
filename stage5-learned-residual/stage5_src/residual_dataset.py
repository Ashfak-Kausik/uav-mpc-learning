"""
PyTorch Dataset wrapper around the combined .npz file from Phase 1.

Each sample is:
  input    : (13,) tensor of (state, control) concatenated
  residual : (9,)  tensor of state_next - state_next_nominal
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ResidualDataset(Dataset):
    def __init__(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        self.state = torch.tensor(data["state"], dtype=torch.float32)
        self.control = torch.tensor(data["control"], dtype=torch.float32)
        self.residual = torch.tensor(data["residual"], dtype=torch.float32)

        if "perturbation_type" in data.files:
            self.perturbation_type = data["perturbation_type"]
            self.perturbation_value = data["perturbation_value"]
        else:
            self.perturbation_type = None
            self.perturbation_value = None

        assert self.state.shape[0] == self.control.shape[0] == self.residual.shape[0], \
            f"Mismatched sizes: state={self.state.shape}, control={self.control.shape}, residual={self.residual.shape}"

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        return {
            "input": torch.cat([self.state[idx], self.control[idx]]),
            "residual": self.residual[idx],
        }

    def all_inputs(self):
        """Return all inputs stacked, for computing normalization stats."""
        return torch.cat([self.state, self.control], dim=1)

    def all_residuals(self):
        return self.residual