"""
Feedforward Residual Controller — pragmatic alternative to in-constraint
residual MPC.

ARCHITECTURE:
  1. The nominal MPC (Stage 3, unchanged) solves with nominal dynamics.
     It produces u_mpc = [T_mpc, wx, wy, wz].
  2. We query the residual model at (x_current, u_mpc) to predict the
     one-step dynamics mismatch.
  3. We convert the predicted state-space residual into a control
     correction (primarily a thrust adjustment) and apply
     u_corrected = u_mpc + correction.

WHY THIS WORKS:
The residual r = [r_px, r_py, r_pz, r_vx, r_vy, r_vz, r_roll, r_pitch, r_yaw]
predicts (actual_next_state - nominal_next_state) over one sim step.
The dominant signal is r_vz, which directly reflects mass and thrust
mismatch effects. If r_vz < 0, the drone is falling more than expected,
so we need extra thrust to compensate. The exact thrust correction is:

    Δthrust = -m * r_vz / dt_sim

derived from: dv_correction = (Δthrust / m) * dt_sim should cancel r_vz.

This sidesteps all interior-point solver issues because the MPC sees
only nominal dynamics and the residual is applied as a feedforward
correction outside the optimization loop.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage3-mpc-baseline"))
if STAGE3_ROOT not in sys.path:
    sys.path.insert(0, STAGE3_ROOT)

from stage3_src.mpc_controller import MPCController
from stage3_src.quadrotor_model_casadi import MASS


class FeedforwardResidualController:
    """
    Wraps a nominal MPCController with a feedforward residual correction.

    The .solve() method matches MPCController's interface so this drops in
    as a controller in the existing test harnesses.
    """

    def __init__(
        self,
        residual_fn,
        sim_dt=0.01,
        mass=MASS,
        thrust_correction_gain=1.0,
        thrust_min=0.0,
        thrust_max=52.0,
        N=20,
        dt=0.05,
        Q_diag=None,
        Q_terminal_scale=10.0,
        R_diag=None,
        rate_max=3.0,
    ):
        """
        Args:
            residual_fn          : CasADi Function (state, control) -> residual.
                                   Built by stage5_src.casadi_residual.
            sim_dt               : the simulator timestep (10 ms). The
                                   residual was trained on this timestep.
            mass                 : nominal mass used in thrust correction.
            thrust_correction_gain
                                 : multiplier on the thrust correction
                                   (1.0 = full correction, 0.5 = half, etc.)
            thrust_min/max       : bounds for the corrected thrust.
            N, dt, Q_diag, ...   : passed through to the underlying
                                   nominal MPC (so existing test scripts
                                   that pass these don't break).
        """
        self.residual_fn = residual_fn
        self.sim_dt = sim_dt
        self.mass = mass
        self.thrust_correction_gain = thrust_correction_gain
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max

        # Underlying nominal MPC — exactly the Stage 3 controller
        self.nominal_mpc = MPCController(
            N=N,
            dt=dt,
            Q_diag=Q_diag,
            Q_terminal_scale=Q_terminal_scale,
            R_diag=R_diag,
            thrust_min=thrust_min,
            thrust_max=thrust_max,
            rate_max=rate_max,
        )

        # Expose N and dt so test scripts can read them like for MPCController
        self.N = self.nominal_mpc.N
        self.dt = self.nominal_mpc.dt

        # Diagnostics — kept for analysis
        self.last_residual = np.zeros(9)
        self.last_thrust_correction = 0.0

    def solve(self, x_current, x_reference):
        """
        Identical interface to MPCController.solve().

        Returns:
            u_corrected : (4,) first control after residual correction.
            X_val       : MPC state trajectory (from nominal MPC).
            U_val       : MPC control trajectory (from nominal MPC).
            status      : 'ok' / 'fallback'.
        """
        # 1. Nominal MPC solve
        u_mpc, X_val, U_val, status = self.nominal_mpc.solve(
            x_current, x_reference)

        # 2. Query the residual at the current state and MPC control
        x_reshape = x_current.reshape(-1, 1)
        u_reshape = u_mpc.reshape(-1, 1)
        residual = np.array(self.residual_fn(x_reshape, u_reshape)).flatten()
        self.last_residual = residual

        # 3. Convert vz residual to thrust correction
        #    residual[5] = r_vz = actual_vz_next - nominal_vz_next over sim_dt
        #    To cancel: Δthrust * sim_dt / mass = -r_vz
        #    => Δthrust = -mass * r_vz / sim_dt
        r_vz = residual[5]
        thrust_correction = (
            -self.mass * r_vz / self.sim_dt * self.thrust_correction_gain
        )

        # Optional: clip the correction to avoid pathological values
        # In practice, |r_vz| ~ 0.01 m/s, so correction ~ 1-2 N.
        thrust_correction = float(np.clip(thrust_correction, -5.0, 5.0))
        self.last_thrust_correction = thrust_correction

        # 4. Apply the correction
        u_corrected = u_mpc.copy()
        u_corrected[0] = np.clip(
            u_corrected[0] + thrust_correction,
            self.thrust_min, self.thrust_max
        )

        return u_corrected, X_val, U_val, status