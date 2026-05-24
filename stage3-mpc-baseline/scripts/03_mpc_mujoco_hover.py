"""
Stage 3 — MPC controlling the Skydio X2 in MuJoCo for hover.

This is the first time the MPC controls the real simulator. The MPC
runs at 50 Hz (its prediction timestep), the body-rate inner loop and
the motor mixer run at every MuJoCo step (100 Hz default), and the
motor mixer output is written to data.ctrl.

The drone should hover at (0, 0, 1) — matching the Stage 2 baseline.
This script is the cleanest way to validate that MPC + MuJoCo works
end-to-end before testing harder trajectories.

Run:
    python3 03_mpc_mujoco_hover.py
    python3 03_mpc_mujoco_hover.py --viewer
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
from stage3_src.trajectory import hover_reference
from stage3_src.quadrotor_model_casadi import (
    quat_to_euler, MASS, GRAVITY,
)

# Stage 2 imports (reused: motor mixer and scene constants).
from stage2_src import x2_constants as C
from stage2_src.cascaded_pd_controller import CascadedPDController


def read_state_for_mpc(data):
    """
    Build the 9-D MPC state vector from MuJoCo's data object.

    MuJoCo provides:
      qpos: 7 elements -> 3 position + 4 quaternion (w, x, y, z)
      qvel: 6 elements -> 3 world-frame linear velocity + 3 body-frame
            angular velocity

    MPC state:
      x = [px, py, pz, vx, vy, vz, roll, pitch, yaw]

    Returns:
        x_mpc : (9,) numpy array
        omega : (3,) body angular velocity (for the inner rate loop)
    """
    p = np.array(data.qpos[0:3])
    q = np.array(data.qpos[3:7])
    v = np.array(data.qvel[0:3])
    omega = np.array(data.qvel[3:6])
    roll, pitch, yaw = quat_to_euler(q)
    x_mpc = np.array([p[0], p[1], p[2],
                      v[0], v[1], v[2],
                      roll, pitch, yaw])
    return x_mpc, omega


def run_mpc_hover(use_viewer=False, duration_s=5.0,
                  p_target=np.array([0.0, 0.0, 1.0])):
    # Load MuJoCo
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)

    # Initial state: drone starts at the hover target, level, at rest.
    # This gives the MPC a smooth initial condition with zero position error.
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # Controllers
    mpc = MPCController()
    rate_ctrl = BodyRateController()
    # We use the Stage 2 controller's motor_mixer method by instantiating
    # the cascaded controller (we only call its motor_mixer).
    mixer = CascadedPDController()

    # Logging
    log_t, log_p, log_v, log_thr, log_solve_ms = [], [], [], [], []
    log_T_cmd, log_omega_des, log_omega = [], [], []

    n_steps = int(duration_s / model.opt.timestep)
    sim_dt = model.opt.timestep

    # Run the MPC every simulator step so the inner loop and optimizer are
    # synchronized at the physics rate.
    k_mpc = 1  # Run MPC every simulator step
    print(f"Simulator dt: {sim_dt*1000:.1f} ms, MPC dt: {mpc.dt*1000:.1f} ms, "
          f"call MPC every {k_mpc} sim steps")

    # Cache of latest MPC output
    last_u_mpc = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

    def step_and_log():
        nonlocal last_u_mpc

        # Build MPC reference: hover at p_target over the horizon
        t_query = data.time + np.arange(mpc.N + 1) * mpc.dt
        x_ref = hover_reference(t_query, p_hover=p_target)

        # Read current MPC state
        x_mpc, omega = read_state_for_mpc(data)

        # Call MPC every k_mpc simulator steps
        if (step_count[0] % k_mpc) == 0:
            t0 = time.time()
            u_mpc, _, _, status = mpc.solve(x_mpc, x_ref)
            solve_ms = (time.time() - t0) * 1000.0
            last_u_mpc = u_mpc
            log_solve_ms.append(solve_ms)
        else:
            solve_ms = np.nan

        # Unpack MPC output
        T_total = float(last_u_mpc[0])
        omega_des = last_u_mpc[1:4]

        # Inner loop: body rates -> body torques
        tau = rate_ctrl.compute(omega_current=omega,
                                omega_desired=omega_des)

        # Motor mixer (from Stage 2)
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
        print(f"Ran {n_steps} sim steps over {duration_s:.2f} s "
              f"(sim time {data.time:.3f} s)")

    log_t = np.array(log_t)
    log_p = np.array(log_p)
    log_v = np.array(log_v)
    log_thr = np.array(log_thr)
    log_solve_ms = np.array(log_solve_ms)
    log_T_cmd = np.array(log_T_cmd)
    log_omega_des = np.array(log_omega_des)
    log_omega = np.array(log_omega)

    # Summary
    final_p = log_p[-1]
    final_err = np.linalg.norm(final_p - p_target)
    print(f"\nTarget position:        {p_target}")
    print(f"Final position:          {final_p}")
    print(f"Final error:             {final_err:.4f} m")
    print(f"Steady-state T command:  {log_T_cmd[-50:].mean():.3f} N "
          f"(theoretical {MASS * GRAVITY:.3f})")
    print(f"Mean motor thrust (last 1 s, per motor): "
          f"{log_thr[-int(1/sim_dt):].mean(axis=0)}")
    print(f"MPC solve time: mean {log_solve_ms.mean():.1f} ms, "
          f"max {log_solve_ms.max():.1f} ms")

    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    for i, label in enumerate(['x', 'y', 'z']):
        axs[0].plot(log_t, log_p[:, i], label=label,
                    color=['C0', 'C1', 'C2'][i])
        axs[0].axhline(p_target[i], ls='--', alpha=0.4,
                       color=['C0', 'C1', 'C2'][i])
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].set_title("MPC + MuJoCo hover")

    for i, label in enumerate(['vx', 'vy', 'vz']):
        axs[1].plot(log_t, log_v[:, i], label=label)
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()

    axs[2].plot(log_t, log_T_cmd, label='T_command', color='black')
    axs[2].axhline(MASS * GRAVITY, ls='--', color='gray', alpha=0.5,
                   label='hover')
    for i in range(4):
        axs[2].plot(log_t, log_thr[:, i], alpha=0.5,
                    label=f'thrust{i+1}')
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

    out_path = os.path.join(HERE, "mpc_mujoco_hover.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true",
                        help="Open the MuJoCo viewer.")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in seconds.")
    args = parser.parse_args()
    run_mpc_hover(use_viewer=args.viewer, duration_s=args.duration)