"""
Stage 2.5: Circle trajectory tracking test.

Commands the drone to fly a horizontal circle of radius R in the
xy-plane at constant altitude. The reference position, velocity, and
acceleration are all available analytically, but the PD controller only
uses the position reference (matching how a naive controller would
behave). The tracking error reveals how much the PD controller
struggles with continuous motion — which is what MPC will improve.

Trajectory:
    x(t) = R * cos(omega * t)
    y(t) = R * sin(omega * t)
    z(t) = z_alt

Run:
    python3 05_circle_test.py            # headless
    python3 05_circle_test.py --viewer   # with viewer
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


# Circle parameters
RADIUS = 1.5            # m
PERIOD = 6.0            # seconds per full revolution
Z_ALT = 1.2             # m
T_RAMP = 2.0            # seconds to fade in the circle (avoids initial jolt)


def get_target(t):
    """
    Reference position, velocity, and acceleration. For t < T_RAMP,
    the radius linearly grows from 0 to RADIUS so the drone smoothly
    enters the circle. After T_RAMP, full circle motion.
    """
    omega = 2.0 * np.pi / PERIOD
    if t < T_RAMP:
        r = RADIUS * t / T_RAMP
        rdot = RADIUS / T_RAMP
        rddot = 0.0
    else:
        r = RADIUS
        rdot = 0.0
        rddot = 0.0

    c, s = np.cos(omega * t), np.sin(omega * t)

    p_des = np.array([r * c, r * s, Z_ALT])
    v_des = np.array([
        rdot * c - r * omega * s,
        rdot * s + r * omega * c,
        0.0,
    ])
    a_des = np.array([
        rddot * c - 2 * rdot * omega * s - r * omega**2 * c,
        rddot * s + 2 * rdot * omega * c - r * omega**2 * s,
        0.0,
    ])
    return p_des, v_des, a_des


def run_circle(use_viewer=False, duration_s=15.0):
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)

    # Start at (R, 0, Z_ALT) — the t = 0 point of the circle.
    p0, _, _ = get_target(0.0)
    data.qpos[0:3] = p0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    controller = CascadedPDController()

    log_t, log_p, log_v, log_thr, log_pdes = [], [], [], [], []

    def step_and_log():
        p = np.array(data.qpos[0:3])
        q = np.array(data.qpos[3:7])
        v = np.array(data.qvel[0:3])
        omega = np.array(data.qvel[3:6])

        p_des, v_des, a_des = get_target(data.time)
        thrusts = controller.compute(p, v, q, omega, p_des, v_des, a_des)
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

    # Tracking error metrics (computed after ramp-in).
    mask = log_t > T_RAMP + 1.0   # exclude ramp + 1s of transient
    err_vec = log_p[mask] - log_pdes[mask]
    err_mag = np.linalg.norm(err_vec, axis=1)
    rms_err = float(np.sqrt(np.mean(err_mag ** 2)))
    max_err = float(np.max(err_mag))

    print(f"\nCircle radius:        {RADIUS:.2f} m")
    print(f"Period:               {PERIOD:.2f} s -> "
          f"angular speed {2*np.pi/PERIOD:.3f} rad/s")
    print(f"Tangential speed:     {RADIUS * 2*np.pi/PERIOD:.3f} m/s")
    print(f"RMS tracking error (after ramp): {rms_err:.4f} m")
    print(f"Max tracking error (after ramp): {max_err:.4f} m")

    # Plot: top view of trajectory + tracking error over time.
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(log_pdes[:, 0], log_pdes[:, 1], '--', alpha=0.5,
             label='reference', color='gray')
    ax1.plot(log_p[:, 0], log_p[:, 1], label='actual', color='C0')
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect('equal')
    ax1.set_title("Top view")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(log_t, err_mag if False else np.linalg.norm(
        log_p - log_pdes, axis=1))
    ax2.axvline(T_RAMP, ls=':', color='gray', alpha=0.5, label='ramp end')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("||position error|| (m)")
    ax2.set_title("Tracking error magnitude")
    ax2.legend()
    ax2.grid(alpha=0.3)

    out_path = os.path.join(HERE, "circle_test.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()
    run_circle(use_viewer=args.viewer, duration_s=args.duration)