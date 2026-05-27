"""
Stage 5 — data collector for residual training.

Runs the Stage 3 nominal MPC on a perturbed plant and logs the per-step
transitions. Each rollout produces:

  state[t]    : (T, 9)  actual state
  control[t]  : (T, 4)  actual control applied (T, wx, wy, wz)
  state_next[t] : (T, 9)  actual next state (from MuJoCo)
  state_next_nominal[t] : (T, 9)  next state predicted by nominal dynamics

The residual target is computed at training time as
  residual = state_next - state_next_nominal

This module reuses everything from Stage 3 (MPC, body-rate loop, mixer)
and Stage 4 (perturbations, scene constants).
"""

import os
import sys
import time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE5_ROOT = os.path.abspath(os.path.join(HERE, ".."))
STAGE4_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage4-perturbations"))
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage3-mpc-baseline"))
STAGE2_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage2-mujoco-setup"))
for p in [STAGE5_ROOT, STAGE4_ROOT, STAGE3_ROOT, STAGE2_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from stage3_src.mpc_controller import MPCController
from stage3_src.body_rate_controller import BodyRateController
from stage3_src.trajectory import circle_reference
from stage3_src.quadrotor_model_casadi import (
    quat_to_euler, make_discrete_dynamics, MASS, GRAVITY,
)
from stage2_src import x2_constants as C
from stage2_src.cascaded_pd_controller import CascadedPDController
from stage4_src.perturbations import (
    apply_mass_perturbation,
    apply_inertia_perturbation,
    LinearDrag,
    MotorLag,
    TimeDelay,
)


CIRCLE_RADIUS = 1.5
CIRCLE_PERIOD = 6.0
CIRCLE_Z_ALT = 1.2
CIRCLE_T_RAMP = 2.0
DEFAULT_DURATION = 15.0


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


def collect_rollout(perturbation_spec, seed=0,
                    duration_s=DEFAULT_DURATION,
                    init_xy_perturb=0.05,
                    init_phase_perturb_rad=np.pi):
    """
    Run a single rollout under the given perturbation and return logged data.

    Args:
        perturbation_spec : dict with "type" and "value" keys.
        seed              : random seed for initial-condition variability.
        duration_s        : rollout duration.
        init_xy_perturb   : max random offset on initial (x, y) in meters.
        init_phase_perturb_rad : max random rotation on circle's starting phase.

    Returns:
        dict with keys:
            "state"               (T, 9)
            "control"             (T, 4)
            "state_next"          (T, 9)
            "state_next_nominal"  (T, 9)
            "perturbation_type"   str
            "perturbation_value"  float or array
            "seed"                int
            "crashed"             bool
            "n_steps"             int
    """
    rng = np.random.default_rng(seed)

    # Fresh MuJoCo model each time
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep

    # Apply perturbation
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
        motor_lag = MotorLag(tau=perturbation_spec["value"], dt=sim_dt,
                             n_motors=4)
        perturbation_info = motor_lag.info()
    elif ptype == "time_delay":
        time_delay = TimeDelay(delay_s=perturbation_spec["value"], dt=sim_dt,
                               n_motors=4,
                               hover_thrust=MASS * GRAVITY / 4.0)
        perturbation_info = time_delay.info()
    elif ptype == "none":
        pass
    else:
        raise ValueError(f"Unknown perturbation type: {ptype}")

    # Initial state: drone exactly at the t=0 reference position
    # (origin due to ramp), level, with small random initial velocity
    # in xy to create state-space variability without breaking feasibility.
    init_velocity_perturb = 0.3
    init_vx = rng.uniform(-init_velocity_perturb, init_velocity_perturb)
    init_vy = rng.uniform(-init_velocity_perturb, init_velocity_perturb)

    # Initial state: drone at the t=0 reference + random offset, level, at rest
    data.qpos[0:3] = [0.0, 0.0, CIRCLE_Z_ALT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[0:3] = [init_vx, init_vy, 0.0]
    mujoco.mj_forward(model, data)

    # Build a phase-shifted reference function
    # circle_reference takes absolute time t; we shift by init_phase
    # by passing t + init_phase / omega as the query time.
    # No phase offset — use the reference as defined.
    time_offset = 0.0

    # Controllers (fresh)
    mpc = MPCController()
    rate_ctrl = BodyRateController()
    mixer = CascadedPDController()

    # Nominal dynamics function (for computing what the unperturbed model
    # would predict at every step).
    nominal_dynamics = make_discrete_dynamics(sim_dt)

    # Logging buffers
    log_state = []
    log_control = []
    log_state_next = []
    log_state_next_nominal = []

    last_u_mpc = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

    n_steps = int(duration_s / sim_dt)
    crashed = False

    for step in range(n_steps):
        # Build reference (with time offset for phase shift)
        t_query = data.time + time_offset + np.arange(mpc.N + 1) * mpc.dt
        x_ref = circle_reference(t_query, radius=CIRCLE_RADIUS,
                                 period=CIRCLE_PERIOD,
                                 z_alt=CIRCLE_Z_ALT,
                                 t_ramp=CIRCLE_T_RAMP)

        # Read current state
        x_mpc, omega = read_state_for_mpc(data)

        # Crash detection
        if (data.qpos[2] < 0.1 or
            np.linalg.norm(data.qpos[0:3]) > 50.0 or
            np.linalg.norm(data.qvel[0:3]) > 30.0):
            crashed = True
            break

        # Solve MPC
        u_mpc, _, _, status = mpc.solve(x_mpc, x_ref)
        last_u_mpc = u_mpc

        T_total = float(last_u_mpc[0])
        omega_des = last_u_mpc[1:4]

        # Save (state, control) BEFORE stepping
        log_state.append(x_mpc.copy())
        log_control.append(last_u_mpc.copy())

        # Compute nominal-dynamics prediction
        x_next_nominal = np.array(
            nominal_dynamics(x_mpc.reshape(-1, 1),
                             last_u_mpc.reshape(-1, 1))
        ).flatten()
        log_state_next_nominal.append(x_next_nominal)

        # Apply control through motor mixer (with perturbations)
        tau_torque = rate_ctrl.compute(omega_current=omega,
                                       omega_desired=omega_des)
        thrusts_commanded = mixer.motor_mixer(T_total, tau_torque)

        if motor_lag is not None:
            thrusts_to_apply = motor_lag.apply(thrusts_commanded)
        elif time_delay is not None:
            thrusts_to_apply = time_delay.apply(thrusts_commanded)
        else:
            thrusts_to_apply = thrusts_commanded

        data.ctrl[:] = thrusts_to_apply

        if drag is not None:
            drag.apply(model, data)

        # Step physics
        mujoco.mj_step(model, data)

        # Read actual next state AFTER stepping
        x_next, _ = read_state_for_mpc(data)
        log_state_next.append(x_next.copy())

    log_state = np.array(log_state)
    log_control = np.array(log_control)
    log_state_next = np.array(log_state_next)
    log_state_next_nominal = np.array(log_state_next_nominal)

    pvalue = perturbation_spec.get("value", None)
    return {
        "state": log_state,
        "control": log_control,
        "state_next": log_state_next,
        "state_next_nominal": log_state_next_nominal,
        "perturbation_type": ptype,
        "perturbation_value": pvalue if pvalue is not None else 0.0,
        "seed": seed,
        "crashed": crashed,
        "n_steps": len(log_state),
    }


def save_rollout(rollout, output_dir):
    """Save a single rollout to .npz file with a sensible filename."""
    ptype = rollout["perturbation_type"]
    pvalue = rollout["perturbation_value"]
    seed = rollout["seed"]

    # Build filename
    if ptype == "none":
        name = f"rollout_nominal_seed{seed}.npz"
    elif ptype == "mass":
        name = f"rollout_mass_{pvalue:.2f}_seed{seed}.npz"
    elif ptype == "drag":
        name = f"rollout_drag_{pvalue:.2f}_seed{seed}.npz"
    elif ptype == "motor_lag":
        name = f"rollout_lag_{int(pvalue*1000)}ms_seed{seed}.npz"
    elif ptype == "time_delay":
        name = f"rollout_delay_{int(pvalue*1000)}ms_seed{seed}.npz"
    else:
        name = f"rollout_{ptype}_seed{seed}.npz"

    out_path = os.path.join(output_dir, name)
    np.savez(out_path,
             state=rollout["state"],
             control=rollout["control"],
             state_next=rollout["state_next"],
             state_next_nominal=rollout["state_next_nominal"],
             perturbation_type=rollout["perturbation_type"],
             perturbation_value=rollout["perturbation_value"],
             seed=rollout["seed"],
             crashed=rollout["crashed"])
    return out_path