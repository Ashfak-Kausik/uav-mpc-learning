"""
Convert a trained PyTorch ResidualMLP into a CasADi Function.

CasADi cannot use PyTorch operations directly inside its symbolic
optimization graph. To use the trained residual inside MPC, we extract
the network's weights, biases, and normalization stats, then rebuild
the forward pass using CasADi primitives (mtimes for linear layers,
ca.tanh for Tanh activations).

The resulting CasADi Function:
  Input  : x (9-dim state), u (4-dim control)
  Output : residual (9-dim correction)

Usage:
    casadi_fn = build_casadi_residual(pytorch_model)
    # In MPC dynamics:
    residual = casadi_fn(state_symbolic, control_symbolic)
    x_next = nominal_dynamics(state, control) + residual

Normalization (input scaling and output scaling) is folded into the
Function automatically. The CasADi Function is fully symbolic and
differentiable; IPOPT can use it inside its constraint Jacobians.

Supported activations: Tanh (default for this project), ReLU (legacy).
Tanh is preferred for MPC integration because it is smooth everywhere;
ReLU's non-smoothness at zero can cause IPOPT to thrash.
"""

import casadi as ca
import torch
import numpy as np


def build_casadi_residual(model, dt=None):
    """
    Build a CasADi Function from a trained ResidualMLP.

    Args:
        model : a trained ResidualMLP, in eval() mode.
        dt    : optional. Unused here; included for future-proofing if
                we want the residual to scale with the timestep.

    Returns:
        ca_fn : a CasADi Function. Call as ca_fn(x_symbolic, u_symbolic)
                to get a symbolic 9-D residual expression.
    """
    model.eval()

    # Extract normalization stats as numpy arrays (float64 for CasADi)
    input_mean = model.input_mean.detach().numpy().astype(np.float64)
    input_std = model.input_std.detach().numpy().astype(np.float64)
    residual_mean = model.residual_mean.detach().numpy().astype(np.float64)
    residual_std = model.residual_std.detach().numpy().astype(np.float64)

    # Walk through the nn.Sequential, recording layer type and params
    layers = []
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.detach().numpy().astype(np.float64)
            b = layer.bias.detach().numpy().astype(np.float64)
            layers.append(("linear", W, b))
        elif isinstance(layer, torch.nn.Tanh):
            layers.append(("tanh", None, None))
        elif isinstance(layer, torch.nn.ReLU):
            layers.append(("relu", None, None))
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    # Build the symbolic forward pass
    x_sym = ca.SX.sym("x_state", 9)
    u_sym = ca.SX.sym("u_control", 4)
    inp = ca.vertcat(x_sym, u_sym)  # (13, 1)

    # Apply input normalization: (inp - mean) / std
    inp_mean_sx = ca.DM(input_mean.reshape(-1, 1))
    inp_std_sx = ca.DM(input_std.reshape(-1, 1))
    inp_norm = (inp - inp_mean_sx) / inp_std_sx

    # Run through layers
    h = inp_norm
    for layer_type, W, b in layers:
        if layer_type == "linear":
            W_sx = ca.DM(W)
            b_sx = ca.DM(b.reshape(-1, 1))
            h = ca.mtimes(W_sx, h) + b_sx
        elif layer_type == "tanh":
            h = ca.tanh(h)
        elif layer_type == "relu":
            h = ca.fmax(h, 0)

    # h is now the normalized residual prediction
    # Squash through tanh to bound the normalized output to (-1, 1) range,
    # then un-normalize. This prevents the residual from producing
    # arbitrarily large corrections when the optimizer explores inputs
    # outside the training distribution.
    h_bounded = ca.tanh(h)
    res_std_sx = ca.DM(residual_std.reshape(-1, 1))
    res_mean_sx = ca.DM(residual_mean.reshape(-1, 1))
    residual = h_bounded * (3.0 * res_std_sx) + res_mean_sx

    # Wrap as a CasADi Function
    ca_fn = ca.Function(
        "residual_fn",
        [x_sym, u_sym],
        [residual],
        ["state", "control"],
        ["residual"],
    )

    return ca_fn


def load_pytorch_model(model_path, state_dim=9, control_dim=4,
                       hidden_dim=64, n_hidden=2):
    """
    Convenience: load a ResidualMLP from a checkpoint file.
    Returns the model in eval mode.
    """
    from stage5_src.residual_model import ResidualMLP

    model = ResidualMLP(state_dim=state_dim, control_dim=control_dim,
                        hidden_dim=hidden_dim, n_hidden=n_hidden)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model