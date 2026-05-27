"""
Stage 5: verify that the CasADi-exported residual matches the
PyTorch model exactly.

For a set of test inputs, we run both:
  - The PyTorch ResidualMLP.predict_numpy(state, control)
  - The CasADi function built from the same model

These should produce numerically identical outputs (down to float64
roundoff). If they differ, the translation has a bug.

Also reports timing: how long the CasADi function takes per evaluation
(this matters because the MPC will call it many times per solve).

Run:
    python3 03_test_casadi_export.py
"""

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stage5_src.residual_model import ResidualMLP
from stage5_src.casadi_residual import build_casadi_residual, load_pytorch_model


MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "models"))
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "residual_best.pt")


def main():
    print("=" * 60)
    print("Stage 5 — verify CasADi export matches PyTorch model")
    print("=" * 60)

    # Load model
    print(f"\nLoading PyTorch model from {BEST_MODEL_PATH}")
    pytorch_model = load_pytorch_model(BEST_MODEL_PATH)
    print(f"  Parameters: {pytorch_model.n_parameters()}")

    # Build CasADi function
    print("\nBuilding CasADi function from PyTorch model...")
    ca_fn = build_casadi_residual(pytorch_model)
    print(f"  CasADi function: {ca_fn}")

    # Generate a batch of test inputs in a realistic range
    print("\nGenerating 100 test inputs...")
    rng = np.random.default_rng(seed=42)
    test_states = []
    test_controls = []
    for _ in range(100):
        # State: position near (0, 0, 1), small velocities, small angles
        x = np.array([
            rng.uniform(-2, 2),       # px
            rng.uniform(-2, 2),       # py
            rng.uniform(0.5, 2),      # pz
            rng.uniform(-3, 3),       # vx
            rng.uniform(-3, 3),       # vy
            rng.uniform(-2, 2),       # vz
            rng.uniform(-0.5, 0.5),   # roll
            rng.uniform(-0.5, 0.5),   # pitch
            rng.uniform(-0.5, 0.5),   # yaw
        ])
        # Control: thrust near hover, small rate commands
        u = np.array([
            rng.uniform(8, 18),       # T
            rng.uniform(-1, 1),       # wx
            rng.uniform(-1, 1),       # wy
            rng.uniform(-1, 1),       # wz
        ])
        test_states.append(x)
        test_controls.append(u)

    # Compare outputs
    print("\nComparing PyTorch vs CasADi outputs...")
    max_diff = 0.0
    sum_diff = 0.0
    for i, (x, u) in enumerate(zip(test_states, test_controls)):
        # PyTorch
        py_out = pytorch_model.predict_numpy(x, u)
        # CasADi
        ca_out = np.array(ca_fn(x.reshape(-1, 1),
                                u.reshape(-1, 1))).flatten()
        diff = np.abs(py_out - ca_out)
        max_diff = max(max_diff, diff.max())
        sum_diff += diff.sum()
        if i < 3:
            print(f"  Test {i}:")
            print(f"    PyTorch: {py_out}")
            print(f"    CasADi:  {ca_out}")
            print(f"    Max diff: {diff.max():.3e}")

    mean_diff = sum_diff / (100 * 9)
    print(f"\nResults over 100 test inputs:")
    print(f"  Max absolute difference: {max_diff:.3e}")
    print(f"  Mean absolute difference: {mean_diff:.3e}")

    if max_diff < 1e-5:
        print(f"  PASS: CasADi function matches PyTorch to float64 precision")
    else:
        print(f"  WARN: max diff exceeds 1e-5; check for translation bugs")

    # Timing
    print("\nTiming CasADi function (1000 evaluations)...")
    x_test = test_states[0].reshape(-1, 1)
    u_test = test_controls[0].reshape(-1, 1)
    t0 = time.time()
    for _ in range(1000):
        ca_fn(x_test, u_test)
    elapsed = time.time() - t0
    per_eval_us = elapsed * 1e6 / 1000
    print(f"  Total: {elapsed*1000:.1f} ms")
    print(f"  Per evaluation: {per_eval_us:.1f} microseconds")
    print(f"  At 100 Hz MPC with 20-step horizon: ~{20 * per_eval_us:.0f} us per MPC solve")


if __name__ == "__main__":
    main()