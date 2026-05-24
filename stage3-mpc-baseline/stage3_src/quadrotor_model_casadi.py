"""
Quadrotor dynamics model in CasADi symbolic form.

This module defines the continuous-time dynamics used by the MPC for
prediction. The state is 9-dimensional (position, velocity, Euler angles)
and the control is 4-dimensional (total thrust + body rates).

The module provides:
  - quadrotor_dynamics(): the symbolic continuous-time dynamics function.
  - rk4_step(): one Runge-Kutta-4 integration step (used inside the MPC
    for shooting-style dynamics constraints).
  - integrate_trajectory(): pure-Python forward integration, used by
    test scripts to verify the dynamics behave physically before being
    handed to the optimizer.

Conventions:
  - Frames: world frame is fixed; z-axis points up.
  - Euler angles: Z-Y-X convention (yaw, pitch, roll), in radians.
  - Body rates: expressed in body frame.
"""

import casadi as ca
import numpy as np

# Physical parameters (must match the Skydio X2 in MuJoCo for the
# nominal-model case).
MASS = 1.325            # kg
GRAVITY = 9.81          # m/s^2

# State and control dimensions.
N_STATE = 9
N_CONTROL = 4


def quadrotor_dynamics(x, u):
    """
    Continuous-time quadrotor dynamics: dx/dt = f(x, u).

    Args:
        x : (9,) CasADi symbolic state [px,py,pz, vx,vy,vz, roll,pitch,yaw]
        u : (4,) CasADi symbolic control [T, wx, wy, wz]

    Returns:
        dx : (9,) CasADi symbolic state derivative
    """
    # Unpack state.
    vx, vy, vz = x[3], x[4], x[5]
    roll, pitch, yaw = x[6], x[7], x[8]

    # Unpack control.
    T = u[0]
    wx, wy, wz = u[1], u[2], u[3]

    # Trig of Euler angles (used several times below).
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cy, sy = ca.cos(yaw), ca.sin(yaw)

    # Body z-axis expressed in world frame (third column of R = Rz Ry Rx).
    z_b_world_x = cy * sp * cr + sy * sr
    z_b_world_y = sy * sp * cr - cy * sr
    z_b_world_z = cp * cr

    # Translational dynamics: a = (T / m) * z_b - g.
    ax = (T / MASS) * z_b_world_x
    ay = (T / MASS) * z_b_world_y
    az = (T / MASS) * z_b_world_z - GRAVITY

    # Euler-angle kinematics from body rates.
    droll  = wx + sr * sp / cp * wy + cr * sp / cp * wz
    dpitch = cr * wy - sr * wz
    dyaw   = sr / cp * wy + cr / cp * wz

    # Assemble derivative.
    dx = ca.vertcat(
        vx, vy, vz,
        ax, ay, az,
        droll, dpitch, dyaw,
    )
    return dx


def rk4_step(x, u, dt):
    """
    One Runge-Kutta-4 integration step.

    More accurate than Euler integration; needed inside the MPC because
    the optimizer is sensitive to integration error.
    """
    k1 = quadrotor_dynamics(x, u)
    k2 = quadrotor_dynamics(x + 0.5 * dt * k1, u)
    k3 = quadrotor_dynamics(x + 0.5 * dt * k2, u)
    k4 = quadrotor_dynamics(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# A CasADi Function wrapping rk4_step for export. The MPC will use this
# as the discrete-time dynamics inside its dynamics constraints.
def make_discrete_dynamics(dt):
    """Build a CasADi Function that maps (x, u) to next-state using RK4."""
    x = ca.SX.sym("x", N_STATE)
    u = ca.SX.sym("u", N_CONTROL)
    x_next = rk4_step(x, u, dt)
    return ca.Function("discrete_dynamics", [x, u], [x_next],
                       ["x", "u"], ["x_next"])


def integrate_trajectory(x0, u_seq, dt):
    """
    Pure-Python (numpy) forward integration of the dynamics.

    Used for testing: given an initial state and a sequence of controls,
    integrates the dynamics forward and returns the state trajectory.
    Internally uses the CasADi-compiled function but evaluates numerically.
    """
    F = make_discrete_dynamics(dt)
    x_traj = [np.array(x0)]
    x = np.array(x0).reshape(-1, 1)
    for u in u_seq:
        u_col = np.array(u).reshape(-1, 1)
        x_next = np.array(F(x, u_col)).flatten()
        x_traj.append(x_next)
        x = x_next.reshape(-1, 1)
    return np.array(x_traj)


def quat_to_euler(q):
    """
    Convert a quaternion (w, x, y, z) to Z-Y-X Euler angles (roll, pitch, yaw).

    Args:
        q : (4,) quaternion in MuJoCo convention (w first)

    Returns:
        (roll, pitch, yaw) in radians

    Note:
        This is a pure-NumPy function; do not use inside CasADi symbolic
        expressions. It is intended for reading state from MuJoCo at
        runtime, so we can hand the MPC its 9-D state vector with Euler
        angles.
    """
    w, x, y, z = q

    # Roll (rotation about x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        # Gimbal lock; pitch = ±π/2
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (rotation about z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
