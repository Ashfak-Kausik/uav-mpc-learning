"""
Stage 3 — side-by-side comparison of PD and MPC on the same circle trajectory.

Runs both controllers in sequence on identical conditions:
  - Skydio X2 in MuJoCo
  - Circle: radius 1.5 m, period 6 s, altitude 1.2 m
  - Smooth cubic ramp over 2 s
  - 15 s total simulation

Produces a single figure with:
  - Top: top-down view of both trajectories overlaid on the reference
  - Bottom: tracking-error magnitude over time for both controllers

Also writes a small CSV with the headline numbers for the paper.

Run:
    python3 06_compare_pd_vs_mpc.py
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

STAGE3_ROOT = os.path.abspath(os.path.join(HERE, ".."))
STAGE2_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage2-mujoco-setup"))
if STAGE3_ROOT not in sys.path:
    sys.path.insert(0, STAGE3_ROOT)
if STAGE2_ROOT not in sys.path:
    sys.path.insert(0, STAGE2_ROOT)

from stage3_src.mpc_controller import MPCController
from stage3_src.body_rate_controller import BodyRateController
from stage3_src.trajectory import circle_reference
from stage3_src.quadrotor_model_casadi import (
    quat_to_euler, MASS, GRAVITY,
)
from stage2_src import x2_constants as C
from stage2_src.cascaded_pd_controller import CascadedPDController


# Circle parameters (must be identical for both runs).
RADIUS = 1.5
PERIOD = 6.0
Z_ALT = 1.2
T_RAMP = 2.0
DURATION = 15.0
ERR_START_T = T_RAMP + 1.0   # exclude ramp + 1s of transient


def smooth_circle_pdref(t):
    """
    Reference position, velocity, acceleration for the PD controller.

    Must use the same smooth cubic ramp as the MPC's trajectory.circle_reference
    so the two controllers are on identical conditions. Returns (p, v, a) for
    a single time t.
    """
    omega = 2.0 * np.pi / PERIOD

    # Smooth cubic ramp factor and its derivative.
    if T_RAMP > 0 and t < T_RAMP:
        tau = t / T_RAMP
        s = 3 * tau ** 2 - 2 * tau ** 3
        sdot = (6 * tau - 6 * tau ** 2) / T_RAMP
        sddot = (6 - 12 * tau) / (T_RAMP ** 2)
    else:
        s, sdot, sddot = 1.0, 0.0, 0.0

    c = np.cos(omega * t)
    sn = np.sin(omega * t)

    # Full-radius circle (ungated).
    px_full = RADIUS * c
    py_full = RADIUS * sn
    vx_full = -RADIUS * omega * sn
    vy_full = RADIUS * omega * c
    ax_full = -RADIUS * (omega ** 2) * c
    ay_full = -RADIUS * (omega ** 2) * sn

    # Apply the ramp.
    p = np.array([s * px_full, s * py_full, Z_ALT])
    v = np.array([sdot * px_full + s * vx_full,
                  sdot * py_full + s * vy_full,
                  0.0])
    a = np.array([sddot * px_full + 2 * sdot * vx_full + s * ax_full,
                  sddot * py_full + 2 * sdot * vy_full + s * ay_full,
                  0.0])
    return p, v, a


def read_state_for_mpc(data):
    p = np.array(data.qpos[0:3])
    q = np.array(data.qpos[3:7])
    v = np.array(data.qvel[0:3])
    omega = np.array(data.qvel[3:6])
    roll, pitch, yaw = quat_to_euler(q)
    x_mpc = np.array([p[0], p[1], p[2],
                      v[0], v[1], v[2],
                      roll, pitch, yaw])
    return x_mpc, omega


# -----------------------------------------------------------------------------
# Runner 1: PD baseline
# -----------------------------------------------------------------------------
def run_pd_baseline(duration_s):
    """Same as Stage 2's circle test, packaged here for direct comparison."""
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep

    # Start at the t=0 reference position (which is origin due to ramp).
    p0, _, _ = smooth_circle_pdref(0.0)
    data.qpos[0:3] = p0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    controller = CascadedPDController()

    log_t, log_p, log_pdes = [], [], []
    n_steps = int(duration_s / sim_dt)

    for _ in range(n_steps):
        p = np.array(data.qpos[0:3])
        q = np.array(data.qpos[3:7])
        v = np.array(data.qvel[0:3])
        omega = np.array(data.qvel[3:6])

        p_des, v_des, a_des = smooth_circle_pdref(data.time)
        thrusts = controller.compute(p, v, q, omega, p_des, v_des, a_des)
        data.ctrl[:] = thrusts
        mujoco.mj_step(model, data)

        log_t.append(data.time)
        log_p.append(p.copy())
        log_pdes.append(p_des.copy())

    return (np.array(log_t),
            np.array(log_p),
            np.array(log_pdes))


# -----------------------------------------------------------------------------
# Runner 2: MPC
# -----------------------------------------------------------------------------
def run_mpc(duration_s):
    """MPC as in 05_mpc_mujoco_circle.py."""
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep

    p0, _, _ = smooth_circle_pdref(0.0)
    data.qpos[0:3] = p0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    mpc = MPCController()
    rate_ctrl = BodyRateController()
    mixer = CascadedPDController()

    log_t, log_p, log_pdes = [], [], []
    log_solve_ms = []
    n_steps = int(duration_s / sim_dt)

    last_u_mpc = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

    for _ in range(n_steps):
        t_query = data.time + np.arange(mpc.N + 1) * mpc.dt
        x_ref = circle_reference(t_query, radius=RADIUS,
                                  period=PERIOD, z_alt=Z_ALT,
                                  t_ramp=T_RAMP)

        x_mpc, omega = read_state_for_mpc(data)

        t0 = time.time()
        u_mpc, _, _, status = mpc.solve(x_mpc, x_ref)
        log_solve_ms.append((time.time() - t0) * 1000.0)
        last_u_mpc = u_mpc

        T_total = float(last_u_mpc[0])
        omega_des = last_u_mpc[1:4]

        tau = rate_ctrl.compute(omega_current=omega,
                                omega_desired=omega_des)
        thrusts = mixer.motor_mixer(T_total, tau)
        data.ctrl[:] = thrusts
        mujoco.mj_step(model, data)

        log_t.append(data.time)
        log_p.append(np.array(data.qpos[0:3]))
        log_pdes.append(x_ref[0:3, 0].copy())

    return (np.array(log_t),
            np.array(log_p),
            np.array(log_pdes),
            np.array(log_solve_ms))


# -----------------------------------------------------------------------------
# Comparison and figure
# -----------------------------------------------------------------------------
def compute_metrics(t, p, pdes, t_start=ERR_START_T):
    mask = t > t_start
    err_full = np.linalg.norm(p - pdes, axis=1)
    err_post = err_full[mask]
    rms = float(np.sqrt(np.mean(err_post ** 2)))
    mx = float(np.max(err_post))
    return rms, mx, err_full


def main():
    print("Running PD baseline...")
    t_pd, p_pd, pdes_pd = run_pd_baseline(DURATION)
    rms_pd, max_pd, err_pd = compute_metrics(t_pd, p_pd, pdes_pd)
    print(f"  PD RMS:  {rms_pd:.4f} m")
    print(f"  PD Max:  {max_pd:.4f} m")

    print("\nRunning MPC...")
    t_mpc, p_mpc, pdes_mpc, solve_ms = run_mpc(DURATION)
    rms_mpc, max_mpc, err_mpc = compute_metrics(t_mpc, p_mpc, pdes_mpc)
    print(f"  MPC RMS: {rms_mpc:.4f} m")
    print(f"  MPC Max: {max_mpc:.4f} m")
    print(f"  MPC solve time mean: {solve_ms.mean():.1f} ms, "
          f"max: {solve_ms.max():.1f} ms")

    print("\n" + "=" * 60)
    print("HEADLINE COMPARISON")
    print("=" * 60)
    print(f"  RMS reduction:   {rms_pd:.4f} -> {rms_mpc:.4f} "
          f"({(rms_pd - rms_mpc) / rms_pd * 100:.1f}% improvement)")
    print(f"  Max reduction:   {max_pd:.4f} -> {max_mpc:.4f} "
          f"({(max_pd - max_mpc) / max_pd * 100:.1f}% improvement)")
    print("=" * 60)

    # CSV with the numbers, for paper-writing later
    csv_path = os.path.join(HERE, "comparison_results.csv")
    with open(csv_path, "w") as f:
        f.write("controller,rms_m,max_m\n")
        f.write(f"PD,{rms_pd:.6f},{max_pd:.6f}\n")
        f.write(f"MPC,{rms_mpc:.6f},{max_mpc:.6f}\n")
    print(f"Numbers written to: {csv_path}")

    # The headline figure.
    fig = plt.figure(figsize=(13, 9))

    # Top-down comparison.
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(pdes_pd[:, 0], pdes_pd[:, 1], '--', alpha=0.5,
             label='reference', color='gray', linewidth=1.5)
    ax1.plot(p_pd[:, 0], p_pd[:, 1], label='PD', color='C0', linewidth=1.5)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect('equal')
    ax1.set_title("PD baseline — top view")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(pdes_mpc[:, 0], pdes_mpc[:, 1], '--', alpha=0.5,
             label='reference', color='gray', linewidth=1.5)
    ax2.plot(p_mpc[:, 0], p_mpc[:, 1], label='MPC', color='C2', linewidth=1.5)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_aspect('equal')
    ax2.set_title("MPC — top view")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Tracking error vs time, both curves overlaid.
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(t_pd, err_pd, label=f"PD (RMS {rms_pd*100:.1f} cm, "
                                  f"max {max_pd*100:.1f} cm)",
             color='C0', linewidth=1.5)
    ax3.plot(t_mpc, err_mpc, label=f"MPC (RMS {rms_mpc*100:.1f} cm, "
                                    f"max {max_mpc*100:.1f} cm)",
             color='C2', linewidth=1.5)
    ax3.axvline(T_RAMP, ls=':', color='gray', alpha=0.5,
                label='ramp end')
    ax3.axvline(ERR_START_T, ls=':', color='black', alpha=0.4,
                label='metric window start')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("||position error|| (m)")
    ax3.set_title("Tracking error magnitude — same trajectory, both controllers")
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3)

    out_path = os.path.join(HERE, "comparison_pd_vs_mpc.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")


if __name__ == "__main__":
    main()