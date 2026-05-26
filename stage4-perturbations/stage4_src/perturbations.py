"""
Perturbation utilities for stage 4.

Each function applies a model-mismatch perturbation to the MuJoCo simulator. 
The MPC's prediction model is unchanged; only the plant is perturbated.
This module is the only place where perturbations semantics are defined.

Three perturbations are implemented:
   - mass_perturbation: increase the drone's mass by a factor (e.g. 1.5x)
   - inertia_perturbation: increase the drone's inertia by a factor (e.g. 1.5x)
   - drag_perturbation: add a simple linear drag force F-drag = -b * v at every step via data.xfrc_applied. The drag coefficient b is a tunable parameter (e.g. 0.5).
"""

import numpy as np 


## Mass perturbation

def apply_mass_perturbation(model, factor, body_name="x2"):
    """
    Increase the mass of body_name by a factor.

    Args:
        model: MuJoCo model
        factor: mass multiplier (e.g. 1.2 for +20% more mass)
        body_name: name of the body to perturb (default "x2")

    Returns:
        info: dict with original_mass, new_mass, factor

    Note: This modifies model.body_mass in place. 
    Call this once at model load time, before running the trial. 
    """
    import mujoco
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in model.")
    
    original_mass = float(model.body_mass[body_id])
    new_mass = original_mass * factor
    model.body_mass[body_id] = new_mass

    return {
        "type": "mass",
        "factor": factor,
        "original_mass": original_mass,
        "new_mass": new_mass,
        "body_name": body_name,
    }


# Inertia perturbation
def apply_inertia_perturbation(model, factor, body_name="x2"):
    """
    Increase the inertia of body_name by a factor.

    Args:
        model: MuJoCo model
        factor: scalar multiplier, or 3-element array for per-axis scaling
        body_name: name of the body to perturb (default "x2")

    Returns:
        info dict.
    """
    import mujoco
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in model.")
    
    factor_array = np.atleast_1d(np.asarray(factor, dtype=float))
    if factor_array.shape == (1,):
        factor_array = np.full(3, factor_array[0])  # uniform scaling
    elif factor_array.shape != (3,):
        raise ValueError(f"Factor must be scalar or 3-element array, got shape {factor_array.shape}")
    
    original_inertia = model.body_inertia[body_id].copy()
    new_inertia = original_inertia * factor_array
    model.body_inertia[body_id] = new_inertia

    return {
        "type": "inertia",
        "factor": factor_array,
        "original_inertia": original_inertia,
        "new_inertia": new_inertia,
        "body_name": body_name,
    }

# Drag perturbation (applied dynamically at each step via data.xfrc_applied)
class LinearDrag:
    """
    Applies F_drag = -b * v at every physics step, where v is the drone's
    linear velocity in world frame and b is the drag coefficient per axis.

    Usage:
        drag = LinearDrag(coefficient=[0.4, 0.4, 0.2])
        # In the simulation loop, before mj_step:
        drag.apply(model, data, body_name="x2")
    """

    def __init__(self, coefficient):
        """
        Args:
            coefficient : scalar (same drag in all 3 axes) or 3-element
                array (per-axis drag coefficients), in units of N*s/m.
        """
        coeff = np.atleast_1d(np.asarray(coefficient, dtype=float))
        if coeff.shape == (1,):
            coeff = np.full(3, coeff[0])
        elif coeff.shape != (3,):
            raise ValueError("coefficient must be scalar or 3-element array")
        self.b = coeff

    def apply(self, model, data, body_name="x2"):
        """Compute and apply drag force at the drone body."""
        import mujoco
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")

        v = data.qvel[0:3]                          # world-frame linear vel
        F_drag = -self.b * v                        # element-wise multiply

        # xfrc_applied[body_id] is a 6-element vector [Fx, Fy, Fz, Tx, Ty, Tz]
        # applied in world frame at the body's CoM.
        data.xfrc_applied[body_id, 0:3] = F_drag
        data.xfrc_applied[body_id, 3:6] = 0.0       # no drag torque

    def info(self):
        return {
            "type": "drag",
            "coefficient": tuple(self.b.tolist()),
        }
    
# Motor lag (first-order filter on each motor command)
class MotorLag:
    """
    First-order filter on each of the four motor commands.

    Models motor/ESC dynamics: T_actual lags T_commanded with time constant tau.
    Update rule per motor per step:
        T_actual <- T_actual + (T_commanded - T_actual) * (dt / tau)
    With tau = 0, T_actual = T_commanded (nominal).
    With tau large, T_actual changes slowly.

    Usage:
        lag = MotorLag(tau=0.05, dt=0.01)  # 50 ms time constant at 100 Hz
        # In the simulation loop:
        actual_thrusts = lag.apply(commanded_thrusts)
        data.ctrl[:] = actual_thrusts
    """

    def __init__(self, tau, dt, n_motors=4):
        """
        Args:
            tau      : time constant in seconds. tau=0 means no lag.
            dt       : simulator timestep (the rate apply() is called at).
            n_motors : number of motors (default 4).
        """
        self.tau = float(tau)
        self.dt = float(dt)
        self.n_motors = n_motors
        self._actual = np.zeros(n_motors)

    def apply(self, commanded):
        """
        Filter the commanded thrusts and return the actual thrusts to apply.

        Args:
            commanded : (n_motors,) commanded thrusts (N)

        Returns:
            actual : (n_motors,) filtered thrusts (N)
        """
        commanded = np.asarray(commanded, dtype=float)
        if self.tau <= 1e-9:
            self._actual = commanded.copy()
        else:
            alpha = self.dt / self.tau
            # Clip alpha to [0, 1] in case dt > tau (unstable region).
            alpha = min(max(alpha, 0.0), 1.0)
            self._actual = self._actual + alpha * (commanded - self._actual)
        return self._actual.copy()

    def reset(self):
        """Reset the filter state. Call between trials."""
        self._actual = np.zeros(self.n_motors)

    def info(self):
        return {
            "type": "motor_lag",
            "tau_s": self.tau,
        }


# Time delay (pure transport delay on the motor command vector)
class TimeDelay:
    """
    Pure transport delay: commands issued at time t reach MuJoCo at t + delay.

    Implemented as a fixed-length queue of past commands. Each call to
    apply() pushes the new command and pops the command that's now ready
    to be applied.

    Usage:
        delay = TimeDelay(delay_s=0.05, dt=0.01)  # 50 ms delay at 100 Hz
        # In the simulation loop:
        actual_thrusts = delay.apply(commanded_thrusts)
        data.ctrl[:] = actual_thrusts
    """

    def __init__(self, delay_s, dt, n_motors=4, hover_thrust=3.25):
        """
        Args:
            delay_s      : delay in seconds.
            dt           : simulator timestep.
            n_motors     : number of motors.
            hover_thrust : thrust value to use when the buffer hasn't filled
                yet (i.e., during the first delay_s seconds). Hover thrust
                is the safe default.
        """
        from collections import deque
        self.delay_s = float(delay_s)
        self.dt = float(dt)
        self.n_motors = n_motors
        self.buffer_len = max(1, int(round(self.delay_s / self.dt)))
        # Pre-fill buffer with hover thrust so initial steps don't get zero.
        initial = np.full(n_motors, hover_thrust)
        self._buffer = deque([initial.copy() for _ in range(self.buffer_len)],
                             maxlen=self.buffer_len)

    def apply(self, commanded):
        """
        Push commanded thrusts into the buffer, pop and return the delayed
        thrusts that are now ready to be applied to MuJoCo.

        Args:
            commanded : (n_motors,) commanded thrusts (N)

        Returns:
            actual : (n_motors,) delayed thrusts (N)
        """
        commanded = np.asarray(commanded, dtype=float).copy()
        if self.delay_s <= 1e-9 or self.buffer_len == 1:
            return commanded
        # Pop the oldest command, push the new one
        actual = self._buffer[0].copy()
        self._buffer.append(commanded)
        return actual

    def reset(self, hover_thrust=3.25):
        """Reset the buffer to hover thrust. Call between trials."""
        from collections import deque
        initial = np.full(self.n_motors, hover_thrust)
        self._buffer = deque([initial.copy() for _ in range(self.buffer_len)],
                             maxlen=self.buffer_len)

    def info(self):
        return {
            "type": "time_delay",
            "delay_s": self.delay_s,
            "delay_steps": self.buffer_len,
        }
    
# Reset Helper
def reset_perturbations(model, data, body_name="x2"):
    """
    Reset all perturbation state. Call this between trials if you reuse
    the same model object.

    Note: This does NOT restore mass/inertia; those must be saved and
    restored by the caller. It only clears xfrc_applied.
    """
    import mujoco
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id >= 0:
        data.xfrc_applied[body_id, :] = 0.0