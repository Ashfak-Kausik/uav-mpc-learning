"""
Experiment runner for Stage 5 feedforward residual evaluation.

Accepts the same perturbation_spec format as Stage 4's experiment_runner,
so all four perturbation types (mass, drag, motor_lag, time_delay) are
supported.

The controller argument lets us run the same trial with either:
  - the nominal MPC (stage3_src.MPCController), or
  - the feedforward residual controller
       (stage5_src.feedforward_residual_controller.FeedforwardResidualController),
without duplicating the loop logic.

Key difference from Stage 4: the runner takes an already-constructed
controller and uses it. Stage 4 constructed a fresh MPCController inside;
we need the residual controller variant, so we let the caller supply it.
"""

import os
import sys
import time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage2-mujoco-setup"))
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage3-mpc-baseline"))
STAGE4_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage4-perturbations"))
for p in [STAGE2_ROOT, STAGE3_ROOT, STAGE4_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from stage2_src import x2_constants as C
from stage2_src.cascaded_pd_controller import CascadedPDController
from stage3_src.body_rate_controller import BodyRateController
from stage3_src.trajectory import circle_reference
from stage3_src.quadrotor_model_casadi import quat_to_euler, MASS, GRAVITY
from stage4_src.perturbations import (
    apply_mass_perturbation,
    apply_inertia_perturbation,
    LinearDrag,
    MotorLag,
    TimeDelay,
)


# Trajectory parameters (matched to Stage 4)
CIRCLE_RADIUS = 1.5
CIRCLE_PERIOD = 6.0
CIRCLE_Z_ALT = 1.2
CIRCLE_T_RAMP = 2.0
DEFAULT_DURATION = 15.0
ERR_START_T = CIRCLE_T_RAMP + 1.0

# Crash thresholds (matched to Stage 4)
CRASH_Z_MIN = 0.1
CRASH_POS_MAX = 50.0
CRASH_VEL_MAX = 30.0


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


def run_trial(controller, perturbation_spec,
              duration_s=DEFAULT_DURATION, verbose=False):
    """
    Run one trial with the given controller and perturbation_spec.

    Args:
        controller          : object with .solve(x, x_ref) and .N/.dt attrs.
                              Either MPCController or FeedforwardResidualController.
        perturbation_spec   : dict matching Stage 4's format:
            {"type": "mass",       "value": float}    # factor
            {"type": "inertia",    "value": float}    # factor
            {"type": "drag",       "value": float}    # b coefficient
            {"type": "motor_lag",  "value": float}    # tau in seconds
            {"type": "time_delay", "value": float}    # delay in seconds
            {"type": "none"}
        duration_s          : sim duration
        verbose             : print per-trial info

    Returns:
        result dict with:
            rms_cm, max_cm, crashed, crash_time,
            solve_ms_mean, solve_ms_max,
            residual_vz_mean, residual_vz_std,
            thrust_correction_mean, thrust_correction_max_abs
            (last four are nan if controller is nominal MPC)
    """
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep

    # Set up perturbation
    drag = None
    motor_lag = None
    time_delay = None

    ptype = perturbation_spec["type"]
    if ptype == "mass":
        apply_mass_perturbation(model, perturbation_spec["value"])
    elif ptype == "inertia":
        apply_inertia_perturbation(model, perturbation_spec["value"])
    elif ptype == "drag":
        drag = LinearDrag(perturbation_spec["value"])
    elif ptype == "motor_lag":
        motor_lag = MotorLag(tau=perturbation_spec["value"],
                             dt=sim_dt, n_motors=4)
    elif ptype == "time_delay":
        time_delay = TimeDelay(delay_s=perturbation_spec["value"],
                                dt=sim_dt, n_motors=4,
                                hover_thrust=MASS * GRAVITY / 4.0)
    elif ptype == "none":
        pass
    else:
        raise ValueError(f"Unknown perturbation type: {ptype}")

    # Initial state at t=0 reference
    data.qpos[0:3] = [0.0, 0.0, CIRCLE_Z_ALT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    rate_ctrl = BodyRateController()
    mixer = CascadedPDController()

    log_t, log_p, log_pdes = [], [], []
    log_solve_ms = []
    log_residual_vz = []
    log_thrust_correction = []

    n_steps = int(duration_s / sim_dt)
    crashed = False
    crash_time = None
    t_start = time.time()

    for step in range(n_steps):
        # Crash detection (check before stepping)
        if (data.qpos[2] < CRASH_Z_MIN or
            np.linalg.norm(data.qpos[0:3]) > CRASH_POS_MAX or
            np.linalg.norm(data.qvel[0:3]) > CRASH_VEL_MAX):
            crashed = True
            crash_time = float(data.time)
            break

        t_query = data.time + np.arange(controller.N + 1) * controller.dt
        x_ref = circle_reference(t_query, radius=CIRCLE_RADIUS,
                                  period=CIRCLE_PERIOD,
                                  z_alt=CIRCLE_Z_ALT,
                                  t_ramp=CIRCLE_T_RAMP)
        x_mpc, omega = read_state_for_mpc(data)

        t0 = time.time()
        u, _, _, status = controller.solve(x_mpc, x_ref)
        log_solve_ms.append((time.time() - t0) * 1000.0)

        if hasattr(controller, "last_residual"):
            log_residual_vz.append(float(controller.last_residual[5]))
            log_thrust_correction.append(
                float(controller.last_thrust_correction))

        T_total = float(u[0])
        omega_des = u[1:4]
        tau = rate_ctrl.compute(omega_current=omega,
                                omega_desired=omega_des)
        thrusts_commanded = mixer.motor_mixer(T_total, tau)

        # Apply motor-side perturbations
        if motor_lag is not None:
            thrusts_to_apply = motor_lag.apply(thrusts_commanded)
        elif time_delay is not None:
            thrusts_to_apply = time_delay.apply(thrusts_commanded)
        else:
            thrusts_to_apply = thrusts_commanded

        data.ctrl[:] = thrusts_to_apply

        # Apply drag if active (before physics step)
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
            "rms_cm": float("nan"),
            "max_cm": float("nan"),
            "crashed": True,
            "crash_time": crash_time,
            "solve_ms_mean": (float(np.mean(log_solve_ms))
                              if log_solve_ms else float("nan")),
            "solve_ms_max": (float(np.max(log_solve_ms))
                              if log_solve_ms else float("nan")),
            "residual_vz_mean": (float(np.mean(log_residual_vz))
                                  if log_residual_vz else float("nan")),
            "residual_vz_std": (float(np.std(log_residual_vz))
                                  if log_residual_vz else float("nan")),
            "thrust_correction_mean": (float(np.mean(log_thrust_correction))
                                        if log_thrust_correction
                                        else float("nan")),
            "thrust_correction_max_abs": (
                float(np.max(np.abs(log_thrust_correction)))
                if log_thrust_correction else float("nan")),
            "elapsed_s": elapsed,
        }
        if verbose:
            print(f"  CRASHED at t={crash_time:.2f}s")
        return result

    log_t = np.array(log_t)
    log_p = np.array(log_p)
    log_pdes = np.array(log_pdes)

    mask = log_t > ERR_START_T
    err = np.linalg.norm(log_p[mask] - log_pdes[mask], axis=1)
    rms_cm = float(np.sqrt(np.mean(err ** 2))) * 100
    max_cm = float(np.max(err)) * 100

    result = {
        "rms_cm": rms_cm,
        "max_cm": max_cm,
        "crashed": False,
        "crash_time": None,
        "solve_ms_mean": float(np.mean(log_solve_ms)),
        "solve_ms_max": float(np.max(log_solve_ms)),
        "residual_vz_mean": (float(np.mean(log_residual_vz))
                              if log_residual_vz else float("nan")),
        "residual_vz_std": (float(np.std(log_residual_vz))
                              if log_residual_vz else float("nan")),
        "thrust_correction_mean": (float(np.mean(log_thrust_correction))
                                    if log_thrust_correction
                                    else float("nan")),
        "thrust_correction_max_abs": (
            float(np.max(np.abs(log_thrust_correction)))
            if log_thrust_correction else float("nan")),
        "elapsed_s": elapsed,
    }

    if verbose:
        print(f"  {ptype}={perturbation_spec.get('value', '-')}: "
              f"RMS={rms_cm:.2f} cm, max={max_cm:.2f} cm, "
              f"time={elapsed:.1f}s")

    return result