"""
Stage 2:4 step-position test.

Starts at hover, then commands a 2 m step in +x. Tests the position
controller's ability to drive the drone laterally and the attitude
controller's ability to coordinate tilt-and-translate.

Run:
    python3 04_step_test.py            # headless
    python3 04_step_test.py --viewer   # with viewer
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if STAGE2_ROOT not in sys.path:
    sys.path.insert(0, STAGE2_ROOT)

from stage2_src import x2_constants as C
from stage2_src.cascaded_pd_controller import CascadedPDController


def get_target(t, t_step=2.0):
    """Hover at (0,0,1) for t < t_step, then jump to (2,0,1)."""
    if t < t_step:
        return np.array([0.0, 0.0, 1.0])
    else:
        return np.array([2.0, 0.0, 1.0])


def run_step(use_viewer=False, duration_s=8.0):
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)

    # Start the drone in the air at the initial hover point.
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    controller = CascadedPDController()

    log_t, log_p, log_v, log_thr, log_pdes = [], [], [], [], []

    def step_and_log():
        p = np.array(data.qpos[0:3])
        q = np.array(data.qpos[3:7])
        v = np.array(data.qvel[0:3])
        omega = np.array(data.qvel[3:6])

        p_des = get_target(data.time)
        thrusts = controller.compute(p, v, q, omega, p_des)
        data.ctrl[:] = thrusts

        mujoco.mj_step(model, data)

        log_t.append(data.time)
        log_p.append(p.copy())
        log_v.append(v.copy())
        log_thr.append(thrusts.copy())
        log_pdes.append(p_des.copy())

    if use_viewer:
        from mujoco import viewer
        with viewer.launch_passive(model, data) as v:
            while v.is_running() and data.time < duration_s:
                step_and_log()
                v.sync()
                time.sleep(max(0.0, model.opt.timestep))
            print(f"Stopped at t = {data.time:.3f} s")
    else:
        n_steps = int(duration_s / model.opt.timestep)
        for _ in range(n_steps):
            step_and_log()
        print(f"Ran {n_steps} steps over {duration_s:.2f} s")

    log_t = np.array(log_t)
    log_p = np.array(log_p)
    log_v = np.array(log_v)
    log_thr = np.array(log_thr)
    log_pdes = np.array(log_pdes)

    # Summary at the end.
    final_p = log_p[-1]
    final_err = np.linalg.norm(final_p - log_pdes[-1])
    print(f"\nFinal target:    {log_pdes[-1]}")
    print(f"Final position:  {final_p}")
    print(f"Final error:     {final_err:.4f} m")
    settling_time = compute_settling_time(log_t, log_p, log_pdes,
                                          tol=0.05, t_step=2.0)
    print(f"5-cm settling time after step: "
          f"{'never' if settling_time is None else f'{settling_time:.2f} s'}")

    # Plot.
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, label in enumerate(['x', 'y', 'z']):
        axs[0].plot(log_t, log_p[:, i], label=label,
                    color=['C0', 'C1', 'C2'][i])
        axs[0].plot(log_t, log_pdes[:, i], '--', alpha=0.5,
                    color=['C0', 'C1', 'C2'][i])
    axs[0].set_ylabel("Position (m)")
    axs[0].legend(loc='upper left')
    axs[0].set_title("Step test — position vs. target (dashed)")

    for i, label in enumerate(['vx', 'vy', 'vz']):
        axs[1].plot(log_t, log_v[:, i], label=label)
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()

    for i in range(4):
        axs[2].plot(log_t, log_thr[:, i], label=f"thrust{i+1}")
    axs[2].axhline(C.HOVER_THRUST_PER_MOTOR, ls='--', color='gray',
                   alpha=0.5, label='hover')
    axs[2].set_ylabel("Thrust (N)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(ncol=2)

    out_path = os.path.join(HERE, "step_test.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


def compute_settling_time(t, p, p_des, tol=0.05, t_step=2.0):
    """First time after t_step that ||p - p_des|| stays under tol."""
    err = np.linalg.norm(p - p_des, axis=1)
    after_step = t >= t_step
    if not np.any(after_step):
        return None
    for i in np.where(after_step)[0]:
        if err[i] < tol and np.all(err[i:] < tol * 2):
            return t[i] - t_step
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()
    run_step(use_viewer=args.viewer, duration_s=args.duration)