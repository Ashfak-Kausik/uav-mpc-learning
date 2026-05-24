"""
Stage 3 — verify the CasADi dynamics model is physically reasonable.

Three tests:
  1. Hover: at hover thrust (T = m*g), zero rates, level orientation,
     the drone should stay still indefinitely.
  2. Free fall: at T = 0, the drone should accelerate downward at g.
  3. Pitch and translate: at hover thrust with a non-zero pitch, the
     drone should accelerate in the corresponding horizontal direction.

If any of these don't match physical intuition, the dynamics module
has a bug and we fix it before building the MPC on top.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if STAGE3_ROOT not in sys.path:
    sys.path.insert(0, STAGE3_ROOT)

from stage3_src.quadrotor_model_casadi import (
    integrate_trajectory,
    MASS,
    GRAVITY,
    N_CONTROL,
)


def test_hover():
    """At hover thrust with zero rates, the drone should not move."""
    print("\n--- Test 1: hover (no motion expected) ---")
    x0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    T_hover = MASS * GRAVITY
    u_seq = [[T_hover, 0.0, 0.0, 0.0]] * 100        # 100 steps
    dt = 0.05
    traj = integrate_trajectory(x0, u_seq, dt)

    final = traj[-1]
    pos_drift = np.linalg.norm(final[0:3] - x0[0:3])
    vel_drift = np.linalg.norm(final[3:6])
    print(f"  Final position: {final[0:3]}")
    print(f"  Final velocity: {final[3:6]}")
    print(f"  Position drift: {pos_drift:.6f} m")
    print(f"  Velocity drift: {vel_drift:.6f} m/s")
    print(f"  Expected: both near 0")
    assert pos_drift < 1e-6, "Hover failed: position drifted"
    assert vel_drift < 1e-6, "Hover failed: velocity grew"
    print("  PASS")


def test_freefall():
    """At T = 0, the drone should accelerate downward at g."""
    print("\n--- Test 2: free fall (T=0) ---")
    x0 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_seq = [[0.0, 0.0, 0.0, 0.0]] * 20            # 1 second
    dt = 0.05
    traj = integrate_trajectory(x0, u_seq, dt)

    # After 1 second, vz should be -9.81 and z should have dropped by g/2.
    print(f"  Final position: {traj[-1, 0:3]}")
    print(f"  Final velocity: {traj[-1, 3:6]}")
    expected_vz = -GRAVITY * 1.0
    expected_z_drop = 0.5 * GRAVITY * 1.0**2
    expected_z = 10.0 - expected_z_drop
    print(f"  Expected vz: {expected_vz:.3f}, got {traj[-1, 5]:.3f}")
    print(f"  Expected z:  {expected_z:.3f}, got {traj[-1, 2]:.3f}")
    # RK4 should be near-exact for this linear case; allow tiny tolerance.
    assert abs(traj[-1, 5] - expected_vz) < 1e-3, "Free-fall vz wrong"
    assert abs(traj[-1, 2] - expected_z) < 1e-3, "Free-fall z wrong"
    print("  PASS")


def test_pitch_translate():
    """
    At hover thrust with pitch = 15 deg, the drone should accelerate
    in the +x direction. The exact magnitude is T*sin(pitch)/m.
    """
    print("\n--- Test 3: pitched at hover thrust (forward translation) ---")
    pitch_deg = 15.0
    pitch_rad = np.deg2rad(pitch_deg)
    x0 = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, pitch_rad, 0.0])
    T_hover = MASS * GRAVITY
    u_seq = [[T_hover, 0.0, 0.0, 0.0]] * 20         # 1 second
    dt = 0.05
    traj = integrate_trajectory(x0, u_seq, dt)

    # Expected horizontal acceleration: a_x = T * sin(pitch) * cos(roll) / m
    # With roll = 0, yaw = 0: a_x = T * sin(pitch) / m.
    a_x_expected = T_hover * np.sin(pitch_rad) / MASS
    # Expected vertical: a_z = T * cos(pitch) / m - g (cos(roll)=1).
    a_z_expected = T_hover * np.cos(pitch_rad) / MASS - GRAVITY

    print(f"  After 1.0 s:")
    print(f"    vx: expected ~{a_x_expected:.3f}, got {traj[-1, 3]:.3f}")
    print(f"    vz: expected ~{a_z_expected:.3f}, got {traj[-1, 5]:.3f}")
    # The drone will accelerate; integration over 1 s gives:
    #   v ~ a * t (since starts at rest, with constant pitch).
    # Should match well.
    assert abs(traj[-1, 3] - a_x_expected) < 0.05, "Pitched vx wrong"
    print("  PASS")


def plot_freefall_trajectory():
    """Visual sanity check: drop the drone from 10 m and plot z(t)."""
    x0 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_seq = [[0.0, 0.0, 0.0, 0.0]] * 30
    dt = 0.05
    traj = integrate_trajectory(x0, u_seq, dt)
    t = np.arange(len(traj)) * dt

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, traj[:, 2], label="simulated z(t)")
    axs[0].plot(t, 10.0 - 0.5 * GRAVITY * t ** 2, '--',
                label="analytical z(t) = 10 - 0.5 g t^2")
    axs[0].set_ylabel("z (m)")
    axs[0].legend()
    axs[0].set_title("Free-fall trajectory: simulated vs. analytical")

    axs[1].plot(t, traj[:, 5], label="simulated vz(t)")
    axs[1].plot(t, -GRAVITY * t, '--',
                label="analytical vz(t) = -g t")
    axs[1].set_ylabel("vz (m/s)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()

    out_path = os.path.join(HERE, "dynamics_freefall.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    print("Testing CasADi quadrotor dynamics model.")
    print("=" * 60)
    test_hover()
    test_freefall()
    test_pitch_translate()
    plot_freefall_trajectory()
    print("\n" + "=" * 60)
    print("All dynamics tests passed.")