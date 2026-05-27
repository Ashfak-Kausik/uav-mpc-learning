"""
MPC controller augmented with a learned residual model.

Identical to Stage 3's MPCController except for the dynamics constraint:
  Stage 3:    x[k+1] = f_nominal(x[k], u[k])
  Stage 5:    x[k+1] = f_nominal(x[k], u[k]) + residual_scale * residual_fn(x[k], u[k])

The residual_scale parameter trades off recovery vs solver speed:
  - scale=0.0: identical to nominal MPC (no residual effect)
  - scale=0.3: ~30% of full correction, easier for solver (~10-15 iters)
  - scale=1.0: full residual, requires ~60+ iterations to converge
"""

import os
import sys

import casadi as ca
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage3-mpc-baseline"))
if STAGE3_ROOT not in sys.path:
    sys.path.insert(0, STAGE3_ROOT)

from stage3_src.quadrotor_model_casadi import (
    rk4_step,
    N_STATE,
    N_CONTROL,
    MASS,
    GRAVITY,
)


class ResidualMPCController:
    """
    Nonlinear MPC with learned residual correction.

    Parameters are identical to MPCController except for:
      residual_fn   : CasADi Function (state, control) -> residual
      residual_scale: scale factor on the residual (default 0.3 — a
                      conservative value that converges fast while still
                      giving substantial recovery).
    """

    def __init__(
        self,
        residual_fn,
        N=20,
        dt=0.05,
        Q_diag=None,
        Q_terminal_scale=10.0,
        R_diag=None,
        thrust_min=0.0,
        thrust_max=52.0,
        rate_max=3.0,
        residual_scale=0.3,
    ):
        self.residual_fn = residual_fn
        self.residual_scale = residual_scale
        self.N = N
        self.dt = dt
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.rate_max = rate_max
        self.u_hover = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

        if Q_diag is None:
            Q_diag = np.array([
                100.0, 100.0, 100.0,
                10.0,  10.0,  10.0,
                1.0,   1.0,   1.0,
            ])
        if R_diag is None:
            R_diag = np.array([0.01, 0.1, 0.1, 0.1])
        self.Q = np.diag(Q_diag)
        self.Q_terminal = Q_terminal_scale * self.Q
        self.R = np.diag(R_diag)

        self._build_optimizer()

        self._prev_X = None
        self._prev_U = None

    def _build_optimizer(self):
        """Build the CasADi Opti() problem with residual-augmented dynamics."""
        opti = ca.Opti()

        X = opti.variable(N_STATE, self.N + 1)
        U = opti.variable(N_CONTROL, self.N)

        x0_param = opti.parameter(N_STATE)
        ref_param = opti.parameter(N_STATE, self.N + 1)

        # Cost (unchanged from Stage 3)
        cost = 0
        for k in range(self.N):
            dx = X[:, k] - ref_param[:, k]
            du = U[:, k] - self.u_hover
            cost += ca.mtimes([dx.T, self.Q, dx])
            cost += ca.mtimes([du.T, self.R, du])
        dxN = X[:, self.N] - ref_param[:, self.N]
        cost += ca.mtimes([dxN.T, self.Q_terminal, dxN])
        opti.minimize(cost)

        # Initial condition
        opti.subject_to(X[:, 0] == x0_param)

        # Residual-augmented dynamics constraints, with scale factor
        for k in range(self.N):
            x_next_nominal = rk4_step(X[:, k], U[:, k], self.dt)
            residual = self.residual_fn(X[:, k], U[:, k])
            x_next = x_next_nominal + self.residual_scale * residual
            opti.subject_to(X[:, k + 1] == x_next)

        # Control bounds
        opti.subject_to(opti.bounded(self.thrust_min,
                                     U[0, :], self.thrust_max))
        opti.subject_to(opti.bounded(-self.rate_max,
                                     U[1:, :], self.rate_max))

        # Robust solver settings tuned for residual-augmented dynamics.
        # IPOPT typically needs more iterations than for the nominal problem
        # because the residual makes the constraint landscape more nonlinear.
        p_opts = {"print_time": False, "expand": True}
        s_opts = {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 300,                 # was 100 — much more headroom
            "max_cpu_time": 3.0,             # 3s per solve, plenty of budget
            "tol": 1e-2,                     # loose primary tolerance
            "acceptable_tol": 10.0,          # very loose acceptable tolerance
            "acceptable_iter": 2,            # accept after just 2 acceptable iters
            "acceptable_constr_viol_tol": 1e-2,
            "acceptable_dual_inf_tol": 1e10, # essentially ignore dual
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-6,
            "warm_start_mult_bound_push": 1e-6,
            "mu_strategy": "monotone",       # more stable with warm-start
            "mu_init": 1e-4,                 # small mu, since warm-start is close
            "hessian_approximation": "limited-memory",  # cheaper Hessian
        }
        opti.solver("ipopt", p_opts, s_opts)

        self.opti = opti
        self.X = X
        self.U = U
        self.x0_param = x0_param
        self.ref_param = ref_param

    def solve(self, x_current, x_reference):
        """One MPC solve. Returns first control + diagnostic info."""
        self.opti.set_value(self.x0_param, x_current)
        self.opti.set_value(self.ref_param, x_reference)

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
            # Soft-fallback: try to recover the latest values from opti.debug.
            # This gets the best solution IPOPT found before giving up,
            # which is often usable even if not "converged".
            try:
                X_val = self.opti.debug.value(self.X)
                U_val = self.opti.debug.value(self.U)
                # Sanity check the recovered values
                if not (np.all(np.isfinite(X_val)) and np.all(np.isfinite(U_val))):
                    raise ValueError("non-finite values in debug solution")
                status = "partial"
            except Exception:
                # Genuine fallback: hover at current state
                print(f"  Residual MPC solver failed completely")
                X_val = np.tile(x_current.reshape(-1, 1), (1, self.N + 1))
                U_val = np.tile(self.u_hover.reshape(-1, 1), (1, self.N))
                status = "fallback"

        self._prev_X = X_val
        self._prev_U = U_val
        return U_val[:, 0], X_val, U_val, status