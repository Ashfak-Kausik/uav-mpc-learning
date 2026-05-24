"""
Stage 3 — MPC + MuJoCo circle trajectory tracking.

Commands the drone to fly a horizontal circle of radius 1.5 m at
1.57 m/s tangential speed, matching the Stage 2 PD baseline. The PD
baseline achieved 2.2 cm RMS / 5.2 cm max tracking error on this same
trajectory; we expect MPC to do significantly better because it
anticipates the curvature instead of reacting to it.

Run:
    python3 05_mpc_mujoco_circle.py
    python3 05_mpc_mujoco_circle.py --viewer
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


# Circle parameters (must match Stage 2's circle test for fair comparison)
RADIUS = 1.5            # m
PERIOD = 6.0            # seconds per full revolution
Z_ALT = 1.2             # m
T_RAMP = 2.0            # seconds to fade in the radius


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


def run_mpc_circle(use_viewer=False, duration_s=15.0):
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)

    # Initial state: at the t=0 position of the circle reference
    # (radius is 0 at t=0 due to the ramp, so drone starts at origin)
    p0 = np.array([0.0, 0.0, Z_ALT])
    data.qpos[0:3] = p0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    mpc = MPCController()
    rate_ctrl = BodyRateController()
    mixer = CascadedPDController()

    log_t, log_p, log_v, log_thr = [], [], [], []
    log_T_cmd, log_omega_des, log_omega = [], [], []
    log_pdes, log_vdes = [], []
    log_solve_ms = []

    n_steps = int(duration_s / model.opt.timestep)
    sim_dt = model.opt.timestep
    k_mpc = 1   # MPC every sim step (100 Hz)

    last_u_mpc = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

    def step_and_log():
        nonlocal last_u_mpc

        # Build MPC reference: circle reference over the prediction horizon
        t_query = data.time + np.arange(mpc.N + 1) * mpc.dt
        x_ref = circle_reference(t_query, radius=RADIUS,
                                  period=PERIOD, z_alt=Z_ALT,
                                  t_ramp=T_RAMP)

        # Read current state
        x_mpc, omega = read_state_for_mpc(data)

        # Solve MPC every k_mpc sim steps
        if (step_count[0] % k_mpc) == 0:
            t0 = time.time()
            u_mpc, _, _, status = mpc.solve(x_mpc, x_ref)
            solve_ms = (time.time() - t0) * 1000.0
            last_u_mpc = u_mpc
            log_solve_ms.append(solve_ms)

        T_total = float(last_u_mpc[0])
        omega_des = last_u_mpc[1:4]

        # Inner loop and motor mixer
        tau = rate_ctrl.compute(omega_current=omega,
                                omega_desired=omega_des)
        thrusts = mixer.motor_mixer(T_total, tau)
        data.ctrl[:] = thrusts

        # Step physics
        mujoco.mj_step(model, data)

        # Log
        log_t.append(data.time)
        log_p.append(np.array(data.qpos[0:3]))
        log_v.append(np.array(data.qvel[0:3]))
        log_thr.append(np.array(thrusts).copy())
        log_T_cmd.append(T_total)
        log_omega_des.append(np.array(omega_des).copy())
        log_omega.append(np.array(omega).copy())
        # Capture the instantaneous reference (first column of x_ref)
        log_pdes.append(x_ref[0:3, 0].copy())
        log_vdes.append(x_ref[3:6, 0].copy())

        step_count[0] += 1

    step_count = [0]

    if use_viewer:
        from mujoco import viewer
        with viewer.launch_passive(model, data) as v:
            while v.is_running() and data.time < duration_s:
                step_and_log()
                v.sync()
                time.sleep(max(0.0, sim_dt))
            print(f"Stopped at t = {data.time:.3f} s")
    else:
        for _ in range(n_steps):
            step_and_log()
        print(f"Ran {n_steps} sim steps over {duration_s:.2f} s")

    log_t = np.array(log_t)
    log_p = np.array(log_p)
    log_v = np.array(log_v)
    log_thr = np.array(log_thr)
    log_solve_ms = np.array(log_solve_ms)
    log_T_cmd = np.array(log_T_cmd)
    log_omega_des = np.array(log_omega_des)
    log_omega = np.array(log_omega)
    log_pdes = np.array(log_pdes)
    log_vdes = np.array(log_vdes)

    # Compute tracking error (exclude ramp + 1 s of transient)
    mask = log_t > T_RAMP + 1.0
    err_vec = log_p[mask] - log_pdes[mask]
    err_mag = np.linalg.norm(err_vec, axis=1)
    rms_err = float(np.sqrt(np.mean(err_mag ** 2)))
    max_err = float(np.max(err_mag))

    print(f"\nCircle radius:        {RADIUS:.2f} m")
    print(f"Period:               {PERIOD:.2f} s -> "
          f"angular speed {2*np.pi/PERIOD:.3f} rad/s")
    print(f"Tangential speed:     {RADIUS * 2*np.pi/PERIOD:.3f} m/s")
    print(f"RMS tracking error:   {rms_err:.4f} m (after ramp+1s)")
    print(f"Max tracking error:   {max_err:.4f} m (after ramp+1s)")
    print(f"MPC solve time: mean {log_solve_ms.mean():.1f} ms, "
          f"max {log_solve_ms.max():.1f} ms")

    # Comparison context: Stage 2 PD result on the same trajectory
    print(f"\n  Stage 2 PD baseline (same trajectory):")
    print(f"    RMS = 0.022 m, max = 0.052 m")
    print(f"  This MPC run:")
    print(f"    RMS = {rms_err:.4f} m, max = {max_err:.4f} m")
    if rms_err < 0.022:
        print(f"  MPC improves RMS by {(0.022 - rms_err) / 0.022 * 100:.1f}%")

    # Plot
    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(log_pdes[:, 0], log_pdes[:, 1], '--', alpha=0.5,
             label='reference', color='gray')
    ax1.plot(log_p[:, 0], log_p[:, 1], label='actual (MPC)', color='C2')
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect('equal')
    ax1.set_title("Top view")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    full_err = np.linalg.norm(log_p - log_pdes, axis=1)
    ax2.plot(log_t, full_err, label='MPC', color='C2')
    ax2.axvline(T_RAMP, ls=':', color='gray', alpha=0.5, label='ramp end')
    ax2.axhline(0.022, ls='--', color='C0', alpha=0.5,
                label='Stage 2 PD RMS (0.022 m)')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("||position error|| (m)")
    ax2.set_title("MPC tracking error magnitude")
    ax2.legend()
    ax2.grid(alpha=0.3)

    out_path = os.path.join(HERE, "mpc_mujoco_circle.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()
    run_mpc_circle(use_viewer=args.viewer, duration_s=args.duration)