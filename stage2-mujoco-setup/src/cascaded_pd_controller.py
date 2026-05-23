"""
Cascaded PD controller for the Skydio X2.

Outer loop: position controller (P/D on position, then desired acceleration
and total thrust).
Inner loop: geometric (rotation-matrix-based) attitude controller (Lee et al.).
Motor mixer: converts total thrust + body torques to four motor commands.

Designed for use with MuJoCo's mj_step loop. The caller passes in the
current state and a desired position; the controller returns the four
motor thrust commands.
"""

import numpy as np
from . import x2_constants as C


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def quat_to_rotmat(q):
    """Convert MuJoCo quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    return R


def vee(S):
    """Extract a 3-vector from a 3x3 skew-symmetric matrix."""
    return 0.5 * np.array([S[2, 1] - S[1, 2],
                            S[0, 2] - S[2, 0],
                            S[1, 0] - S[0, 1]])


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------
class CascadedPDController:
    """
    Cascaded position + attitude PD controller.

    Gains are tunable; the defaults below are a reasonable starting point
    for the Skydio X2 at 100 Hz control. Expect to tune.
    """

    def __init__(
        self,
        kp_pos=np.array([6.0, 6.0, 8.0]),       # position P (per axis)
        kd_pos=np.array([4.0, 4.0, 5.0]),       # position D (per axis)
        kp_att=np.array([20.0, 20.0, 4.0]),     # attitude P (roll, pitch, yaw)
        kd_att=np.array([2.0, 2.0, 0.5]),       # attitude D
        yaw_des=0.0,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.yaw_des = yaw_des  # commanded yaw (rad); 0 = facing +x

    # -------------------------------------------------------------------------
    # Outer loop: position → desired acceleration → total thrust + R_des
    # -------------------------------------------------------------------------
    def position_loop(self, p, v, p_des, v_des=None, a_des_ff=None):
        """
        Compute desired total thrust magnitude and desired body z-axis.

        Args:
            p        : (3,) current position
            v        : (3,) current velocity
            p_des    : (3,) desired position
            v_des    : (3,) desired velocity (defaults to zero)
            a_des_ff : (3,) feed-forward desired acceleration
                       (defaults to zero -- pure PD without feed-forward)

        Returns:
            T_total : float, desired total thrust (N)
            z_des   : (3,) unit vector, desired body z-axis (world frame)
            a_des   : (3,) desired acceleration (world frame), for logging
        """
        if v_des is None:
            v_des = np.zeros(3)
        if a_des_ff is None:
            a_des_ff = np.zeros(3)

        e_p = p_des - p          # position error
        e_v = v_des - v          # velocity error
        a_des = (
            self.kp_pos * e_p
            + self.kd_pos * e_v
            + a_des_ff
            + np.array([0.0, 0.0, C.GRAVITY])
        )
        T_total = C.MASS * np.linalg.norm(a_des)
        z_des = a_des / max(np.linalg.norm(a_des), 1e-6)
        return T_total, z_des, a_des

    def desired_rotation(self, z_des):
        """
        Construct R_des from desired z-axis and commanded yaw.

        We define an "intermediate frame" rotated about world z by yaw_des,
        whose x-axis is x_c = [cos(yaw), sin(yaw), 0]. Then:
            y_des = (z_des x x_c) / ||z_des x x_c||
            x_des = y_des x z_des
        and R_des = [x_des | y_des | z_des].
        """
        c, s = np.cos(self.yaw_des), np.sin(self.yaw_des)
        x_c = np.array([c, s, 0.0])
        y_des = np.cross(z_des, x_c)
        y_des = y_des / max(np.linalg.norm(y_des), 1e-6)
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack([x_des, y_des, z_des])
        return R_des

    # -------------------------------------------------------------------------
    # Inner loop: R, omega → desired body torques
    # -------------------------------------------------------------------------
    def attitude_loop(self, R, omega, R_des):
        """
        Geometric attitude controller (Lee et al. 2010 style).

        Returns:
            tau : (3,) desired body torques (Nm) in body frame
        """
        S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = vee(S_err)            # rotation error (body frame)
        e_omega = omega             # angular velocity error (target = 0)
        tau = -self.kp_att * e_R - self.kd_att * e_omega
        return tau

    # -------------------------------------------------------------------------
    # Motor mixer: T_total, tau → four motor thrusts
    # -------------------------------------------------------------------------
    def motor_mixer(self, T_total, tau):
        """
        Invert the X-configuration mixing to compute per-motor thrusts.

        Layout (from site positions in the X2 model):
            motor 1: (-x, -y), drag CW   (kappa negative)
            motor 2: (-x, +y), drag CCW  (kappa positive)
            motor 3: (+x, +y), drag CW   (kappa negative)
            motor 4: (+x, -y), drag CCW  (kappa positive)

        Mixing (forward direction):
            T   = T1 + T2 + T3 + T4
            tau_x = ARM_Y * (-T1 + T2 + T3 - T4)   [roll]
            tau_y = ARM_X * ( T1 + T2 - T3 - T4)   [pitch]
            tau_z = KAPPA * (-T1 + T2 - T3 + T4)   [yaw]

        Inverting:
        """
        T_each = T_total / 4.0
        rx = tau[0] / (4.0 * C.ARM_Y)   # roll contribution
        ry = tau[1] / (4.0 * C.ARM_X)   # pitch contribution
        rz = tau[2] / (4.0 * C.KAPPA)   # yaw contribution

        T1 = T_each - rx + ry - rz
        T2 = T_each + rx + ry + rz
        T3 = T_each + rx - ry - rz
        T4 = T_each - rx - ry + rz

        thrusts = np.clip([T1, T2, T3, T4], C.THRUST_MIN, C.THRUST_MAX)
        return thrusts

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def compute(self, p, v, q, omega, p_des, v_des=None, a_des_ff=None):
        """
        One-shot controller call.

        Args:
            p        : (3,) world-frame position
            v        : (3,) world-frame linear velocity
            q        : (4,) MuJoCo quaternion (w, x, y, z)
            omega    : (3,) body-frame angular velocity
            p_des    : (3,) desired position
            v_des    : (3,) desired velocity (optional; defaults to zero)
            a_des_ff : (3,) feed-forward desired acceleration (optional)

        Returns:
            thrusts : (4,) motor commands in Newtons
        """
        R = quat_to_rotmat(q)
        T_total, z_des, _ = self.position_loop(p, v, p_des, v_des, a_des_ff)
        R_des = self.desired_rotation(z_des)
        tau = self.attitude_loop(R, omega, R_des)
        thrusts = self.motor_mixer(T_total, tau)
        return thrusts