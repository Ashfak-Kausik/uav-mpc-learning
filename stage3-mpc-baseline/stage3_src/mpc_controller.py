"""
Model Predictive Controller for quadrotor trajectory tracking.

Position-MPC formulation with kinematic-thrust cascade:
  - State: 9-D [position, velocity, Euler angles]
  - Control: 4-D [total thrust, body roll rate, body pitch rate, body yaw rate]
  - Horizon: 20 steps at 0.05 s = 1 second look-ahead
  - Solver: IPOPT via CasADi

The MPC is called at the simulator's control rate. It returns the first
control of the optimized horizon, which is then passed to the body-rate
inner loop (defined separately).
"""

import casadi as ca
import numpy as np

from .quadrotor_model_casadi import (
    rk4_step,
    N_STATE,
    N_CONTROL,
    MASS,
    GRAVITY,
)


class MPCController:
    """
    Nonlinear MPC for quadrotor trajectory tracking.

    Tunable hyperparameters live in __init__. The solver is set up once
    when the controller is constructed. At runtime, call solve() with the
    current state and a reference trajectory of length N+1.
    """

    def __init__(
        self,
        N=20,                            # prediction horizon (steps)
        dt=0.05,                         # prediction timestep (s)
        Q_diag=None,                     # state cost weights
        Q_terminal_scale=10.0,           # terminal cost = scale * Q
        R_diag=None,                     # control cost weights
        thrust_min=0.0,
        thrust_max=52.0,                 # 4 motors x 13 N
        rate_max=3.0,                    # rad/s
    ):
        self.N = N
        self.dt = dt
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.rate_max = rate_max
        self.u_hover = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

        # Default cost weights. Position is weighted heavily, velocity
        # moderately, orientation lightly. Controls are weighted small to
        # let the optimizer use authority freely.
        if Q_diag is None:
            Q_diag = np.array([
                100.0, 100.0, 100.0,     # position
                10.0,  10.0,  10.0,      # velocity
                1.0,   1.0,   1.0,       # roll, pitch, yaw
            ])
        if R_diag is None:
            R_diag = np.array([
                0.01,                     # thrust
                0.1, 0.1, 0.1,            # body rates
            ])
        self.Q = np.diag(Q_diag)
        self.Q_terminal = Q_terminal_scale * self.Q
        self.R = np.diag(R_diag)

        # Build the optimization problem.
        self._build_optimizer()

        # Warm-start storage.
        self._prev_X = None
        self._prev_U = None

    def _build_optimizer(self):
        """Construct the CasADi Opti() problem once at init."""
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(N_STATE, self.N + 1)
        U = opti.variable(N_CONTROL, self.N)

        # Parameters (set at runtime)
        x0_param = opti.parameter(N_STATE)
        ref_param = opti.parameter(N_STATE, self.N + 1)

        # Cost
        cost = 0
        for k in range(self.N):
            dx = X[:, k] - ref_param[:, k]
            du = U[:, k] - self.u_hover
            cost += ca.mtimes([dx.T, self.Q, dx])
            cost += ca.mtimes([du.T, self.R, du])
        # Terminal cost
        dxN = X[:, self.N] - ref_param[:, self.N]
        cost += ca.mtimes([dxN.T, self.Q_terminal, dxN])
        opti.minimize(cost)

        # Initial condition
        opti.subject_to(X[:, 0] == x0_param)

        # Dynamics constraints (RK4 shooting)
        for k in range(self.N):
            x_next = rk4_step(X[:, k], U[:, k], self.dt)
            opti.subject_to(X[:, k + 1] == x_next)

        # Control bounds
        opti.subject_to(opti.bounded(self.thrust_min,
                                     U[0, :], self.thrust_max))
        opti.subject_to(opti.bounded(-self.rate_max,
                                     U[1:, :], self.rate_max))

        # Solver settings: silent IPOPT with relaxed convergence and better warm-starting.
        p_opts = {"print_time": False, "expand": True}
        s_opts = {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 500,
            "tol": 1e-3,
            "acceptable_tol": 1e-1,
            "acceptable_iter": 5,
            "warm_start_init_point": "yes",
            "mu_strategy": "adaptive",
        }
        opti.solver("ipopt", p_opts, s_opts)

        # Save for use in solve()
        self.opti = opti
        self.X = X
        self.U = U
        self.x0_param = x0_param
        self.ref_param = ref_param

    def solve(self, x_current, x_reference):
        """
        Solve the MPC for one step.

        Args:
            x_current   : (9,) current state
            x_reference : (9, N+1) reference trajectory over horizon

        Returns:
            u0       : (4,) first control of the optimized horizon
            X_pred   : (9, N+1) predicted state trajectory (for logging)
            U_pred   : (4, N)   predicted control trajectory (for logging)
            status   : str, "ok" if solver succeeded
        """
        self.opti.set_value(self.x0_param, x_current)
        self.opti.set_value(self.ref_param, x_reference)

        # Warm-start from previous solution if available
        if self._prev_X is not None:
            # Shift previous solution by one step
            X_init = np.zeros_like(self._prev_X)
            X_init[:, :-1] = self._prev_X[:, 1:]
            X_init[:, -1] = self._prev_X[:, -1]
            U_init = np.zeros_like(self._prev_U)
            U_init[:, :-1] = self._prev_U[:, 1:]
            U_init[:, -1] = self._prev_U[:, -1]
            self.opti.set_initial(self.X, X_init)
            self.opti.set_initial(self.U, U_init)
        else:
            # First call: seed with the reference trajectory and hover control.
            # This gives IPOPT a feasible-ish starting point near the solution.
            X_init = np.array(x_reference)
            U_init = np.tile(self.u_hover.reshape(-1, 1), (1, self.N))
            self.opti.set_initial(self.X, X_init)
            self.opti.set_initial(self.U, U_init)

        try:
            sol = self.opti.solve()
            X_val = sol.value(self.X)
            U_val = sol.value(self.U)
            status = "ok"
        except RuntimeError as e:
            # Solver failed; fall back to hover thrust and zero rates
            # so the drone doesn't tumble out of control.
            print(f"  MPC solver failed: {e}")
            X_val = np.tile(x_current.reshape(-1, 1), (1, self.N + 1))
            U_val = np.tile(self.u_hover.reshape(-1, 1), (1, self.N))
            status = "fallback"

        self._prev_X = X_val
        self._prev_U = U_val
        return U_val[:, 0], X_val, U_val, status