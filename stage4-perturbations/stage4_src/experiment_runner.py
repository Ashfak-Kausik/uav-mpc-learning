"""
Experiment runner for Stage 4.

Encapsulates the "run an MPC trial on a perturbed simulator, return metrics"
pattern. Each Stage 4 sweep script is a thin loop over this function.
"""

import os
import sys
import time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE4_ROOT = os.path.abspath(os.path.join(HERE, ".."))
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage3-mpc-baseline"))
STAGE2_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage2-mujoco-setup"))
for p in [STAGE4_ROOT, STAGE3_ROOT, STAGE2_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from stage3_src.mpc_controller import MPCController
from stage3_src.body_rate_controller import BodyRateController
from stage3_src.trajectory import circle_reference
from stage3_src.quadrotor_model_casadi import (
    quat_to_euler, MASS, GRAVITY,
)
from stage2_src import x2_constants as C
from stage2_src.cascaded_pd_controller import CascadedPDController
from stage4_src.perturbations import (
    apply_mass_perturbation,
    apply_inertia_perturbation,
    LinearDrag,
    MotorLag,
    TimeDelay,
    reset_perturbations,
)


CIRCLE_RADIUS = 1.5
CIRCLE_PERIOD = 6.0
CIRCLE_Z_ALT = 1.2
CIRCLE_T_RAMP = 2.0
DEFAULT_DURATION = 15.0
ERR_START_T = CIRCLE_T_RAMP + 1.0


def read_state_for_mpc(data):
    p = np.array(data.qpos[0:3])
    q = np.array(data.qpos[3:7])
    v = np.array(data.qvel[0:3])
    omega = np.array(data.qvel[3:6])
    roll, pitch, yaw = quat_to_euler(q)
    x_mpc = np.array([p[0], p[1], p[2],
                      v[0], v[1], v[2],
                      roll, pitch, yaw])
    return x_mpc, omega


def run_trial(perturbation_spec, duration_s=DEFAULT_DURATION,
              verbose=False):
    """
    Run one MPC trial with the given perturbation.

    Args:
        perturbation_spec : dict with keys
            "type"  : one of "mass", "inertia", "drag", "none"
            "value" : scalar or 3-vector depending on type
        duration_s        : simulation duration in seconds
        verbose           : print per-trial info to stdout

    Returns:
        result : dict with keys:
            "rms_err"            : RMS tracking error (m), or np.inf if crashed
            "max_err"            : max tracking error (m), or np.inf if crashed
            "stable"             : bool, True if drone didn't lose stability
            "final_pos"          : (3,) final position
            "perturbation_info"  : the info dict from the perturbation
    """
    # Fresh model each trial
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep

    # Apply perturbation. Some types modify the model once at load (mass,
    # inertia); some apply forces during the loop (drag); some filter the
    # motor commands (motor_lag, time_delay).
    perturbation_info = {"type": "none"}
    drag = None
    motor_lag = None
    time_delay = None

    ptype = perturbation_spec["type"]
    if ptype == "mass":
        perturbation_info = apply_mass_perturbation(model,
                                                     perturbation_spec["value"])
    elif ptype == "inertia":
        perturbation_info = apply_inertia_perturbation(model,
                                                        perturbation_spec["value"])
    elif ptype == "drag":
        drag = LinearDrag(perturbation_spec["value"])
        perturbation_info = drag.info()
    elif ptype == "motor_lag":
        motor_lag = MotorLag(tau=perturbation_spec["value"],
                             dt=sim_dt,
                             n_motors=4)
        perturbation_info = motor_lag.info()
    elif ptype == "time_delay":
        # Use per-motor hover thrust to pre-fill the delay buffer.
        time_delay = TimeDelay(delay_s=perturbation_spec["value"],
                               dt=sim_dt,
                               n_motors=4,
                               hover_thrust=MASS * GRAVITY / 4.0)
        perturbation_info = time_delay.info()
    elif ptype == "none":
        pass
    else:
        raise ValueError(f"Unknown perturbation type: {ptype}")

    # Initial state: at the t=0 reference (drone at origin due to ramp)
    data.qpos[0:3] = [0.0, 0.0, CIRCLE_Z_ALT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # Controllers (fresh each trial, so warm-start doesn't carry over)
    mpc = MPCController()
    rate_ctrl = BodyRateController()
    mixer = CascadedPDController()

    log_t, log_p, log_pdes = [], [], []
    n_steps = int(duration_s / sim_dt)

    # Stability heuristics
    crashed = False
    last_u_mpc = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

    t_start = time.time()
    for step in range(n_steps):
        t_query = data.time + np.arange(mpc.N + 1) * mpc.dt
        x_ref = circle_reference(t_query, radius=CIRCLE_RADIUS,
                                  period=CIRCLE_PERIOD,
                                  z_alt=CIRCLE_Z_ALT,
                                  t_ramp=CIRCLE_T_RAMP)

        x_mpc, omega = read_state_for_mpc(data)

        # Crash detection: if drone fell or flew way off, declare crash
        if (data.qpos[2] < 0.1 or
            np.linalg.norm(data.qpos[0:3]) > 50.0 or
            np.linalg.norm(data.qvel[0:3]) > 30.0):
            crashed = True
            break

        u_mpc, _, _, status = mpc.solve(x_mpc, x_ref)
        last_u_mpc = u_mpc

        T_total = float(last_u_mpc[0])
        omega_des = last_u_mpc[1:4]

        tau = rate_ctrl.compute(omega_current=omega,
                                omega_desired=omega_des)
        thrusts_commanded = mixer.motor_mixer(T_total, tau)

        # Apply motor-side perturbations to the thrust commands before
        # writing them to MuJoCo. Only one of motor_lag or time_delay
        # is active in a given trial.
        if motor_lag is not None:
            thrusts_to_apply = motor_lag.apply(thrusts_commanded)
        elif time_delay is not None:
            thrusts_to_apply = time_delay.apply(thrusts_commanded)
        else:
            thrusts_to_apply = thrusts_commanded

        data.ctrl[:] = thrusts_to_apply

        # Apply drag if active (before the physics step)
        if drag is not None:
            drag.apply(model, data)

        mujoco.mj_step(model, data)

        log_t.append(data.time)
        log_p.append(np.array(data.qpos[0:3]))
        log_pdes.append(x_ref[0:3, 0].copy())

    elapsed = time.time() - t_start

    # Compute metrics
    if crashed:
        result = {
            "rms_err": float("inf"),
            "max_err": float("inf"),
            "stable": False,
            "final_pos": np.array(data.qpos[0:3]).copy(),
            "perturbation_info": perturbation_info,
            "elapsed_s": elapsed,
        }
        if verbose:
            print(f"  CRASHED at t={data.time:.2f}s, "
                  f"pos={data.qpos[0:3]}, vel={data.qvel[0:3]}")
        return result

    log_t = np.array(log_t)
    log_p = np.array(log_p)
    log_pdes = np.array(log_pdes)

    mask = log_t > ERR_START_T
    err_full = np.linalg.norm(log_p - log_pdes, axis=1)
    err_post = err_full[mask]
    rms = float(np.sqrt(np.mean(err_post ** 2)))
    mx = float(np.max(err_post))

    result = {
        "rms_err": rms,
        "max_err": mx,
        "stable": True,
        "final_pos": log_p[-1].copy(),
        "perturbation_info": perturbation_info,
        "elapsed_s": elapsed,
    }

    if verbose:
        print(f"  {perturbation_info}: RMS={rms*100:.2f} cm, "
              f"max={mx*100:.2f} cm, time={elapsed:.1f}s")

    return result