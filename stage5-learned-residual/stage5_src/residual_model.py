"""
Residual model: small MLP that predicts the dynamics residual.

Input  : (state, control) concatenated, 13-dim
Output : 9-dim residual on (px, py, pz, vx, vy, vz, roll, pitch, yaw)

Architecture choices (important for MPC integration):
  - 2 hidden layers, 64 neurons each (~5,000 parameters)
  - Tanh activation throughout. Tanh is smooth everywhere, which is
    essential for IPOPT (interior-point solvers cannot reliably handle
    non-smooth constraint functions like ReLU's kink at zero).
  - Linear output layer.
  - Input/output normalization stored as buffers so they save/load with
    the model.

Why tanh and not ReLU: when the MPC optimizer differentiates through
the residual function inside its dynamics constraints, ReLU's
non-smoothness causes IPOPT to thrash near kinks and fail with
'Maximum_Iterations_Exceeded'. Tanh is smooth and well-behaved for
gradient-based solvers.

Why input std clamping: training data on circle trajectories has
extremely narrow distributions on some state dimensions (e.g., yaw
std ~0.008, pz std ~0.027). Dividing by these tiny std values
amplifies the input gradient on those dimensions by 100x+, creating
an ill-conditioned constraint Jacobian (condition number ~1e6).
Clamping std to a minimum of 0.1 bounds the amplification at 10x and
keeps IPOPT solvable.
"""

import torch
import torch.nn as nn
import numpy as np


class ResidualMLP(nn.Module):
    def __init__(self, state_dim=9, control_dim=4, hidden_dim=64,
                 n_hidden=2):
        super().__init__()
        in_dim = state_dim + control_dim
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

        # Normalization stats (set by fit_normalization after data is loaded).
        # Registered as buffers so they save/load with the model.
        self.register_buffer("input_mean", torch.zeros(in_dim))
        self.register_buffer("input_std",  torch.ones(in_dim))
        self.register_buffer("residual_mean", torch.zeros(state_dim))
        self.register_buffer("residual_std",  torch.ones(state_dim))

        self.state_dim = state_dim
        self.control_dim = control_dim

    def fit_normalization(self, inputs, residuals, min_input_std=0.1):
        """
        Compute and store mean and std of inputs and residuals.

        Input std is clamped to a minimum (default 0.1) to prevent
        pathological amplification on narrow-distribution dimensions
        (yaw, pz, vz, wz on circle data). Without clamping, the
        optimizer's constraint Jacobian becomes ill-conditioned and
        IPOPT cannot solve.

        Args:
            inputs        : (N, state_dim + control_dim) torch tensor
            residuals     : (N, state_dim) torch tensor
            min_input_std : minimum value for input std (default 0.1)
        """
        eps = 1e-6
        self.input_mean = inputs.mean(dim=0)
        raw_std = inputs.std(dim=0) + eps
        if min_input_std is not None:
            self.input_std = torch.clamp(raw_std, min=min_input_std)
        else:
            self.input_std = raw_std
        self.residual_mean = residuals.mean(dim=0)
        self.residual_std = residuals.std(dim=0) + eps

    def forward(self, inputs):
        """
        Forward pass with normalization in/out.

        Args:
            inputs : (B, 13) raw (state, control) — NOT pre-normalized

        Returns:
            residual : (B, 9) raw residual prediction — NOT pre-normalized
        """
        # Normalize input
        x = (inputs - self.input_mean) / self.input_std
        # Network forward (linear -> tanh -> linear -> tanh -> linear)
        y_norm = self.net(x)
        # Un-normalize output
        residual = y_norm * self.residual_std + self.residual_mean
        return residual

    def predict_numpy(self, state, control):
        """
        Convenience for inference outside the training loop.

        Args:
            state   : (9,) numpy array
            control : (4,) numpy array

        Returns:
            residual : (9,) numpy array
        """
        with torch.no_grad():
            sc = torch.cat([torch.tensor(state, dtype=torch.float32),
                            torch.tensor(control, dtype=torch.float32)])
            residual = self.forward(sc.unsqueeze(0)).squeeze(0)
            return residual.numpy()

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)