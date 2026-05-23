"""
Stage 2 — hover test for the cascaded PD controller.

Starts the X2 just above the ground, commands it to hover at (0, 0, 1) m,
and runs for a fixed duration. Logs position over time and plots the
result. The viewer can be enabled or disabled via a flag.

Run:
    python3 03_hover_test.py           # headless, logs + plots
    python3 03_hover_test.py --viewer  # opens viewer
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt

# Make the src module importable.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from src import x2_constants as C
from src.cascaded_pd_controller import CascadedPDController


def run_hover(use_viewer=False, duration_s=5.0, p_des=np.array([0.0, 0.0, 1.0])):
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)

    # Move the drone slightly above the ground to start.
    data.qpos[0:3] = [0.0, 0.0, 0.3]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]    # identity quaternion (w,x,y,z)
    mujoco.mj_forward(model, data)

    controller = CascadedPDController()

    log_t, log_p, log_v, log_thr = [], [], [], []

    def step_and_log():
        # Read state.
        p = np.array(data.qpos[0:3])
        q = np.array(data.qpos[3:7])
        v = np.array(data.qvel[0:3])
        omega = np.array(data.qvel[3:6])

        # Compute control.
        thrusts = controller.compute(p, v, q, omega, p_des)
        data.ctrl[:] = thrusts

        # Step physics.
        mujoco.mj_step(model, data)

        # Log.
        log_t.append(data.time)
        log_p.append(p.copy())
        log_v.append(v.copy())
        log_thr.append(thrusts.copy())

    if use_viewer:
        from mujoco import viewer
        with viewer.launch_passive(model, data) as v:
            t0 = time.time()
            while v.is_running() and data.time < duration_s:
                step_and_log()
                v.sync()
                time.sleep(max(0.0, model.opt.timestep))
            print(f"Stopped at t = {data.time:.3f} s")
    else:
        n_steps = int(duration_s / model.opt.timestep)
        for _ in range(n_steps):
            step_and_log()
        print(f"Ran {n_steps} steps over {duration_s:.2f} s "
              f"(sim time {data.time:.3f} s)")

    log_t = np.array(log_t)
    log_p = np.array(log_p)
    log_v = np.array(log_v)
    log_thr = np.array(log_thr)

    # Summary.
    final_p = log_p[-1]
    final_err = np.linalg.norm(final_p - p_des)
    print(f"\nTarget position:  {p_des}")
    print(f"Final position:    {final_p}")
    print(f"Final error:       {final_err:.4f} m")
    print(f"Mean thrust per motor (last 1 s): "
          f"{log_thr[-int(1/model.opt.timestep):].mean(axis=0)} N")
    print(f"Hover thrust (theoretical): "
          f"{C.HOVER_THRUST_PER_MOTOR:.3f} N per motor")

    # Plot.
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, label in enumerate(['x', 'y', 'z']):
        axs[0].plot(log_t, log_p[:, i], label=f'{label}')
        axs[0].axhline(p_des[i], ls='--', alpha=0.4,
                       color=['C0', 'C1', 'C2'][i])
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].set_title("Hover test — position vs. time")

    for i, label in enumerate(['vx', 'vy', 'vz']):
        axs[1].plot(log_t, log_v[:, i], label=label)
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()

    for i in range(4):
        axs[2].plot(log_t, log_thr[:, i], label=f"thrust{i+1}")
    axs[2].axhline(C.HOVER_THRUST_PER_MOTOR, ls='--',
                   color='gray', alpha=0.5, label='hover (theoretical)')
    axs[2].set_ylabel("Thrust (N)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(ncol=2)

    out_path = os.path.join(HERE, "hover_test.png")
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
    run_hover(use_viewer=args.viewer, duration_s=args.duration)