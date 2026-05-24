"""
Stage 3 — MPC + MuJoCo step test.

Drone starts at (0, 0, 1) hovering. At t = 1 s, the target jumps to
(2, 0, 1). Tests whether MPC can plan a clean lateral step without
overshooting or saturating motors prolongedly.

Compare to Stage 2's step test: PD took 1.60 s to settle within 5 cm,
with motors briefly saturating to 13 N. MPC should match or beat
that, and ideally show less saturation because it plans ahead.

Run:
    python3 04_mpc_mujoco_step.py
    python3 04_mpc_mujoco_step.py --viewer
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Add Stage 2 and Stage 3 project roots to the path so we can import
# their respective packages (stage2_src, stage3_src). No name collisions
# because the package names differ.
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, ".."))
STAGE2_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage2-mujoco-setup"))
if STAGE3_ROOT not in sys.path:
    sys.path.insert(0, STAGE3_ROOT)
if STAGE2_ROOT not in sys.path:
    sys.path.insert(0, STAGE2_ROOT)

# Stage 3 imports.
from stage3_src.mpc_controller import MPCController
from stage3_src.body_rate_controller import BodyRateController
from stage3_src.trajectory import step_reference
from stage3_src.quadrotor_model_casadi import (
    quat_to_euler, MASS, GRAVITY,
)

# Stage 2 imports (reused: motor mixer and scene constants).
from stage2_src import x2_constants as C
from stage2_src.cascaded_pd_controller import CascadedPDController


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


def compute_settling_time(t, p, p_target, t_step, tol=0.05):
    """First time after t_step that ||p - p_target|| stays under tol."""
    err = np.linalg.norm(p - p_target, axis=1)
    after_step = t >= t_step
    if not np.any(after_step):
        return None
    idx = np.where(after_step)[0]
    for i in idx:
        if err[i] < tol and np.all(err[i:] < tol * 2):
            return t[i] - t_step
    return None


def run_mpc_step(use_viewer=False, duration_s=6.0):
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)

    # Initial state: hovering at the start of the step trajectory
    p_before = np.array([0.0, 0.0, 1.0])
    p_after = np.array([2.0, 0.0, 1.0])
    t_step = 1.0

    data.qpos[0:3] = p_before
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    mpc = MPCController()
    rate_ctrl = BodyRateController()
    mixer = CascadedPDController()

    log_t, log_p, log_v, log_thr = [], [], [], []
    log_T_cmd, log_omega_des, log_omega = [], [], []
    log_solve_ms = []

    n_steps = int(duration_s / model.opt.timestep)
    sim_dt = model.opt.timestep
    k_mpc = 1   # MPC every sim step (100 Hz)

    last_u_mpc = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

    def step_and_log():
        nonlocal last_u_mpc

        t_query = data.time + np.arange(mpc.N + 1) * mpc.dt
        x_ref = step_reference(t_query, t_step=t_step,
                               p_before=p_before, p_after=p_after)

        x_mpc, omega = read_state_for_mpc(data)

        if (step_count[0] % k_mpc) == 0:
            t0 = time.time()
            u_mpc, _, _, status = mpc.solve(x_mpc, x_ref)
            solve_ms = (time.time() - t0) * 1000.0
            last_u_mpc = u_mpc
            log_solve_ms.append(solve_ms)

        T_total = float(last_u_mpc[0])
        omega_des = last_u_mpc[1:4]

        tau = rate_ctrl.compute(omega_current=omega,
                                omega_desired=omega_des)
        thrusts = mixer.motor_mixer(T_total, tau)
        data.ctrl[:] = thrusts

        mujoco.mj_step(model, data)

        log_t.append(data.time)
        log_p.append(np.array(data.qpos[0:3]))
        log_v.append(np.array(data.qvel[0:3]))
        log_thr.append(np.array(thrusts).copy())
        log_T_cmd.append(T_total)
        log_omega_des.append(np.array(omega_des).copy())
        log_omega.append(np.array(omega).copy())

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

    final_p = log_p[-1]
    final_err = np.linalg.norm(final_p - p_after)
    settling = compute_settling_time(log_t, log_p, p_after,
                                     t_step=t_step, tol=0.05)
    print(f"\nTarget after step:        {p_after}")
    print(f"Final position:            {final_p}")
    print(f"Final error:               {final_err:.4f} m")
    print(f"5-cm settling time:        "
          f"{'never' if settling is None else f'{settling:.2f} s'}")
    print(f"Max motor thrust observed: {log_thr.max():.2f} N")
    print(f"MPC solve time: mean {log_solve_ms.mean():.1f} ms, "
          f"max {log_solve_ms.max():.1f} ms")

    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    p_target_traj = np.array([p_before if t < t_step else p_after
                              for t in log_t])
    for i, label in enumerate(['x', 'y', 'z']):
        axs[0].plot(log_t, log_p[:, i], label=label,
                    color=['C0', 'C1', 'C2'][i])
        axs[0].plot(log_t, p_target_traj[:, i], '--', alpha=0.4,
                    color=['C0', 'C1', 'C2'][i])
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].set_title("MPC + MuJoCo step test")

    for i, label in enumerate(['vx', 'vy', 'vz']):
        axs[1].plot(log_t, log_v[:, i], label=label)
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()

    axs[2].plot(log_t, log_T_cmd, label='T_cmd', color='black')
    for i in range(4):
        axs[2].plot(log_t, log_thr[:, i], alpha=0.5,
                    label=f'thrust{i+1}')
    axs[2].axhline(13.0, ls=':', color='red', alpha=0.5,
                   label='per-motor max')
    axs[2].set_ylabel("Thrust (N)")
    axs[2].legend(ncol=3, fontsize='small')

    for i, label in enumerate(['ωx', 'ωy', 'ωz']):
        axs[3].plot(log_t, log_omega_des[:, i], '--',
                    color=['C0', 'C1', 'C2'][i],
                    label=f"{label}_des", alpha=0.7)
        axs[3].plot(log_t, log_omega[:, i],
                    color=['C0', 'C1', 'C2'][i],
                    label=f"{label}", alpha=0.7)
    axs[3].set_ylabel("Body rates (rad/s)")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend(ncol=3, fontsize='small')

    out_path = os.path.join(HERE, "mpc_mujoco_step.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()
    run_mpc_step(use_viewer=args.viewer, duration_s=args.duration)