"""
Diagnose why IPOPT struggles with the residual.

Runs ONE solve with print_level=5 so IPOPT tells us what it's doing.
Tests three regimes:
  1. Residual scaled by 0.01 (essentially nominal)
  2. Residual scaled by 0.1
  3. Residual scaled by 1.0 (the failing case)
"""
import os
import sys
import time
import numpy as np
import casadi as ca

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "stage3-mpc-baseline"))
if STAGE3_ROOT not in sys.path:
    sys.path.insert(0, STAGE3_ROOT)

from stage3_src.quadrotor_model_casadi import rk4_step, MASS, GRAVITY, N_STATE, N_CONTROL
from stage3_src.trajectory import circle_reference
from stage5_src.casadi_residual import build_casadi_residual, load_pytorch_model

MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "models"))
model = load_pytorch_model(os.path.join(MODELS_DIR, "residual_best.pt"))
residual_fn = build_casadi_residual(model)

print("=" * 60)
print("Step 1: Check residual values at a realistic MPC trajectory")
print("=" * 60)
test_traj = circle_reference(np.arange(21) * 0.05)  # 1s of reference
u_hover = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])
print(f"{'k':<4} {'pos':<25} {'vel':<25} {'residual vz':<12}")
for k in range(21):
    x = test_traj[:, k]
    u = u_hover
    res = np.array(residual_fn(x.reshape(-1,1), u.reshape(-1,1))).flatten()
    p_str = f"({x[0]:5.2f},{x[1]:5.2f},{x[2]:5.2f})"
    v_str = f"({x[3]:5.2f},{x[4]:5.2f},{x[5]:5.2f})"
    print(f"{k:<4} {p_str:<25} {v_str:<25} {res[5]:>10.6f}")

print(f"\nMax |residual| over trajectory: {np.abs([np.array(residual_fn(test_traj[:, k].reshape(-1,1), u_hover.reshape(-1,1))).flatten() for k in range(21)]).max():.6f}")

print("\n" + "=" * 60)
print("Step 2: Try MPC with varying residual scale, verbose IPOPT")
print("=" * 60)

N = 20; dt_mpc = 0.05
Q = np.diag([100.,100.,100., 10.,10.,10., 1.,1.,1.])
QT = 10 * Q
R = np.diag([0.01, 0.1, 0.1, 0.1])

def build_and_solve(scale, verbose=False):
    opti = ca.Opti()
    X = opti.variable(N_STATE, N + 1)
    U = opti.variable(N_CONTROL, N)
    x0 = opti.parameter(N_STATE)
    ref = opti.parameter(N_STATE, N + 1)
    cost = 0
    for k in range(N):
        dx = X[:, k] - ref[:, k]
        du = U[:, k] - u_hover
        cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])
    cost += ca.mtimes([(X[:, N] - ref[:, N]).T, QT, X[:, N] - ref[:, N]])
    opti.minimize(cost)
    opti.subject_to(X[:, 0] == x0)
    for k in range(N):
        xn = rk4_step(X[:, k], U[:, k], dt_mpc)
        if scale > 0:
            xn = xn + scale * residual_fn(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == xn)
    opti.subject_to(opti.bounded(0.0, U[0, :], 52.0))
    opti.subject_to(opti.bounded(-3.0, U[1:, :], 3.0))
    s_opts = {"print_level": 5 if verbose else 0,
              "sb": "yes",
              "max_iter": 100, "max_cpu_time": 5.0,
              "tol": 1e-2, "acceptable_tol": 1.0,
              "warm_start_init_point": "yes",
              "mu_strategy": "adaptive"}
    opti.solver("ipopt", {"print_time": False, "expand": True}, s_opts)
    opti.set_value(x0, np.array([0.,0.,1.2, 0.,0.,0., 0.,0.,0.]))
    opti.set_value(ref, test_traj)
    opti.set_initial(X, test_traj)
    opti.set_initial(U, np.tile(u_hover.reshape(-1,1), (1, N)))
    t0 = time.time()
    try:
        sol = opti.solve()
        elapsed = time.time() - t0
        iters = sol.stats()["iter_count"]
        print(f"  scale={scale}: OK in {elapsed:.2f}s, {iters} iterations")
        return True
    except RuntimeError as e:
        elapsed = time.time() - t0
        status = "unknown"
        if "Maximum_Iterations" in str(e):
            status = "max_iter"
        elif "Maximum_CpuTime" in str(e):
            status = "timeout"
        print(f"  scale={scale}: FAILED ({status}) in {elapsed:.2f}s")
        return False

print("\nNominal (scale=0):")
build_and_solve(0.0)
print("\nTiny residual (scale=0.01):")
build_and_solve(0.01)
print("\nSmall residual (scale=0.1):")
build_and_solve(0.1)
print("\nMedium residual (scale=0.5):")
build_and_solve(0.5)
print("\nFull residual (scale=1.0):")
build_and_solve(1.0)

print("\n" + "=" * 60)
print("Step 3: Verbose solve for full residual to see what IPOPT does")
print("=" * 60)
build_and_solve(1.0, verbose=True)