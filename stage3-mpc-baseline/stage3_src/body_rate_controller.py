"""
Inner-loop body-rate controller.

Receives desired body rates (omega_x_des, omega_y_des, omega_z_des) from
the MPC and the current body rates from the simulator, and produces body
torques. The motor mixer (reused from Stage 2) then converts (T, torques)
into four motor commands.

This is a simple proportional controller on rate error. A more
sophisticated version could add integral action, but for our purposes
(running the inner loop at 100 Hz matching the simulator's physics
rate), pure P is sufficient.
"""

import numpy as np


class BodyRateController:
    """
    Proportional controller on body rates.

    Gains are per-axis, in units of (N*m) / (rad/s).

    The default gains are tuned for the Skydio X2's inertia:
      Ixx ~= 0.061, Iyy ~= 0.036, Izz ~= 0.025 kg*m^2
    A rule of thumb is K_rate = bandwidth * inertia, where bandwidth is
    in rad/s. With bandwidth ~ 100 rad/s (i.e. ~16 Hz response), we get
    K_rate ~= (6, 4, 2.5) for the three axes. We use slightly different
    values here, tuned empirically.
    """

    def __init__(
        self,
        kp_rate=np.array([8.0, 5.0, 2.0]),  # roll, pitch, yaw gains
    ):
        self.kp_rate = kp_rate

    def compute(self, omega_current, omega_desired):
        """
        Compute body torques from rate error.

        Args:
            omega_current : (3,) current body angular velocity (rad/s)
            omega_desired : (3,) desired body angular velocity (rad/s)

        Returns:
            tau : (3,) body torques (N*m)
        """
        e_omega = omega_desired - omega_current
        tau = self.kp_rate * e_omega
        return tau