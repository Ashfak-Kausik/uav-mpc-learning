"""
Stage 3 — first MPC test: hover at (0, 0, 1).

Runs the MPC inside a Python loop (no MuJoCo yet — we use the same
CasADi-compiled dynamics for the "plant"). This isolates whether the
MPC formulation works before we worry about the simulator interface.

Once this works, we'll plug it into MuJoCo via the inner loop in
Step 3.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from src.mpc_controller import MPCController
from src.trajectory import hover_reference
from src.quadrotor_model_casadi import (
    make_discrete_dynamics, MASS, GRAVITY, N_STATE,
)


def run_hover_mpc(duration_s=5.0, dt_sim=0.02):
    """
    Simulate MPC hover for duration_s seconds.

    The 'plant' is our own CasADi dynamics (not MuJoCo yet). This lets us
    test the MPC in isolation. The MPC plans at 50 Hz (dt_mpc = 0.05 s)
    while the plant runs at 50 Hz too (dt_sim = 0.02 s would be 50 Hz,
    here we use 0.02 = 50 Hz; could go finer if needed).
    """
    mpc = MPCController()
    dt_mpc = mpc.dt
    plant_F = make_discrete_dynamics(dt_sim)

    # Initial state: drone at 0.3 m, level, at rest
    x = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    log_t, log_x, log_u, log_solve_ms = [], [], [], []

    t = 0.0
    n_steps = int(duration_s / dt_sim)
    print(f"Running MPC hover for {duration_s:.1f} s "
          f"({n_steps} simulator steps, MPC dt = {dt_mpc:.3f} s)")

    for step in range(n_steps):
        # Build reference: hover at (0, 0, 1) for the next N+1 steps
        t_query = t + np.arange(mpc.N + 1) * dt_mpc
        x_ref = hover_reference(t_query)

        # Solve MPC
        t0 = time.time()
        u, X_pred, U_pred, status = mpc.solve(x, x_ref)
        solve_ms = (time.time() - t0) * 1000.0

        # Apply first control, step plant forward
        x_next = np.array(plant_F(x.reshape(-1, 1),
                                  u.reshape(-1, 1))).flatten()

        # Log
        log_t.append(t)
        log_x.append(x.copy())
        log_u.append(u.copy())
        log_solve_ms.append(solve_ms)

        x = x_next
        t += dt_sim

    log_t = np.array(log_t)
    log_x = np.array(log_x)
    log_u = np.array(log_u)
    log_solve_ms = np.array(log_solve_ms)

    # Summary
    final_pos = log_x[-1, 0:3]
    final_err = np.linalg.norm(final_pos - np.array([0.0, 0.0, 1.0]))
    print(f"\nFinal position: {final_pos}")
    print(f"Final error:    {final_err:.6f} m")
    print(f"Steady-state thrust: {log_u[-50:, 0].mean():.3f} N "
          f"(theoretical: {MASS * GRAVITY:.3f} N)")
    print(f"MPC solve time: mean {log_solve_ms.mean():.1f} ms, "
          f"max {log_solve_ms.max():.1f} ms")

    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for i, label in enumerate(['x', 'y', 'z']):
        axs[0].plot(log_t, log_x[:, i], label=label,
                    color=['C0', 'C1', 'C2'][i])
    axs[0].axhline(1.0, ls='--', color='gray', alpha=0.4, label='z target')
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].set_title("MPC hover — position vs. time")

    for i, label in enumerate(['vx', 'vy', 'vz']):
        axs[1].plot(log_t, log_x[:, 3 + i], label=label)
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()

    axs[2].plot(log_t, log_u[:, 0], label='T (total thrust)', color='black')
    axs[2].axhline(MASS * GRAVITY, ls='--', color='gray', alpha=0.5,
                   label='hover thrust')
    axs[2].set_ylabel("Thrust (N)")
    axs[2].legend()

    axs[3].plot(log_t, log_solve_ms)
    axs[3].set_ylabel("MPC solve time (ms)")
    axs[3].set_xlabel("Time (s)")

    out_path = os.path.join(HERE, "mpc_hover.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    run_hover_mpc(duration_s=5.0, dt_sim=0.02)