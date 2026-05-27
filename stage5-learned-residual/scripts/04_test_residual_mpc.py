"""
Stage 5: first end-to-end test of residual-augmented MPC.

Runs three trials on the perturbed plant (mass factor 1.2):
  1. Nominal MPC (Stage 3 controller, no residual)
  2. Residual MPC (Stage 5 controller, with learned residual)
  3. Nominal MPC on nominal plant (no perturbation, for reference)

Compares tracking error across all three. The expected pattern:
  - Nominal MPC + perturbation: degraded (around 5 cm RMS for mass=1.2)
  - Residual MPC + perturbation: improved, closer to nominal performance
  - Nominal MPC + no perturbation: baseline (around 1.65 cm RMS)

If the residual is working, trial 2 should sit between trials 1 and 3,
with most of the gap closed.

Run:
    python3 04_test_residual_mpc.py
"""

import os
import sys
import time

import numpy as np
import mujoco
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

STAGE2_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage2-mujoco-setup"))
STAGE3_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage3-mpc-baseline"))
STAGE4_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage4-perturbations"))
for p in [STAGE2_ROOT, STAGE3_ROOT, STAGE4_ROOT]:
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
from stage4_src.perturbations import apply_mass_perturbation

from stage5_src.casadi_residual import build_casadi_residual, load_pytorch_model
from stage5_src.mpc_with_residual import ResidualMPCController


MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "models"))
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "residual_best.pt")

CIRCLE_RADIUS = 1.5
CIRCLE_PERIOD = 6.0
CIRCLE_Z_ALT = 1.2
CIRCLE_T_RAMP = 2.0
DURATION = 15.0
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


def run_one_trial(controller, mass_factor=None, label="trial"):
    """
    Run a single circle trajectory with the given controller, optionally
    applying a mass perturbation. Returns logged trajectory and metrics.
    """
    model = mujoco.MjModel.from_xml_path(C.SCENE_PATH)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep

    if mass_factor is not None:
        apply_mass_perturbation(model, mass_factor)

    data.qpos[0:3] = [0.0, 0.0, CIRCLE_Z_ALT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    rate_ctrl = BodyRateController()
    mixer = CascadedPDController()

    log_t, log_p, log_pdes = [], [], []
    log_solve_ms = []
    n_steps = int(DURATION / sim_dt)
    last_u = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])

    print(f"\n  Running {label}...")
    t_start = time.time()

    for step in range(n_steps):
        t_query = data.time + np.arange(controller.N + 1) * controller.dt
        x_ref = circle_reference(t_query, radius=CIRCLE_RADIUS,
                                  period=CIRCLE_PERIOD,
                                  z_alt=CIRCLE_Z_ALT,
                                  t_ramp=CIRCLE_T_RAMP)
        x_mpc, omega = read_state_for_mpc(data)

        t0 = time.time()
        u, _, _, status = controller.solve(x_mpc, x_ref)
        log_solve_ms.append((time.time() - t0) * 1000.0)
        last_u = u

        T_total = float(last_u[0])
        omega_des = last_u[1:4]

        tau = rate_ctrl.compute(omega_current=omega,
                                omega_desired=omega_des)
        thrusts = mixer.motor_mixer(T_total, tau)
        data.ctrl[:] = thrusts
        mujoco.mj_step(model, data)

        log_t.append(data.time)
        log_p.append(np.array(data.qpos[0:3]))
        log_pdes.append(x_ref[0:3, 0].copy())

    elapsed = time.time() - t_start

    log_t = np.array(log_t)
    log_p = np.array(log_p)
    log_pdes = np.array(log_pdes)
    log_solve_ms = np.array(log_solve_ms)

    mask = log_t > ERR_START_T
    err = np.linalg.norm(log_p[mask] - log_pdes[mask], axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    mx = float(np.max(err))

    print(f"    Done. RMS={rms*100:.2f} cm, Max={mx*100:.2f} cm")
    print(f"    Wall time: {elapsed:.1f} s, MPC solve: mean "
          f"{log_solve_ms.mean():.1f} ms, max {log_solve_ms.max():.1f} ms")

    return {
        "label": label,
        "t": log_t,
        "p": log_p,
        "pdes": log_pdes,
        "err": np.linalg.norm(log_p - log_pdes, axis=1),
        "rms_cm": rms * 100,
        "max_cm": mx * 100,
        "solve_ms_mean": log_solve_ms.mean(),
        "solve_ms_max": log_solve_ms.max(),
    }


def main():
    print("=" * 60)
    print("Stage 5 — residual MPC verification")
    print("=" * 60)

    # Load PyTorch model and build CasADi residual
    print(f"\nLoading residual model from {BEST_MODEL_PATH}")
    pytorch_model = load_pytorch_model(BEST_MODEL_PATH)
    ca_residual = build_casadi_residual(pytorch_model)
    print("Built CasADi residual function.")

    # Build controllers
    nominal_mpc = MPCController()
    residual_mpc = ResidualMPCController(residual_fn=ca_residual)

    # Run trials
    PERTURB_MASS = 1.2

    trial_nom_nom = run_one_trial(
        nominal_mpc, mass_factor=None,
        label=f"Nominal MPC + nominal plant (baseline)"
    )

    trial_nom_perturbed = run_one_trial(
        MPCController(),       # fresh, to avoid warm-start contamination
        mass_factor=PERTURB_MASS,
        label=f"Nominal MPC + plant mass={PERTURB_MASS}"
    )

    trial_res_perturbed = run_one_trial(
        residual_mpc, mass_factor=PERTURB_MASS,
        label=f"Residual MPC + plant mass={PERTURB_MASS}"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Trial':<45} {'RMS (cm)':>10} {'Max (cm)':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10}")
    for trial in [trial_nom_nom, trial_nom_perturbed, trial_res_perturbed]:
        print(f"  {trial['label']:<45} {trial['rms_cm']:>10.2f} {trial['max_cm']:>10.2f}")

    # Compute the closure
    gap = trial_nom_perturbed['rms_cm'] - trial_nom_nom['rms_cm']
    recovered = trial_nom_perturbed['rms_cm'] - trial_res_perturbed['rms_cm']
    if gap > 0:
        pct = recovered / gap * 100
        print(f"\n  Performance gap (perturbed - nominal): {gap:.2f} cm")
        print(f"  Recovered by residual:                  {recovered:.2f} cm "
              f"({pct:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for trial, color in [(trial_nom_nom, "C2"),
                          (trial_nom_perturbed, "C3"),
                          (trial_res_perturbed, "C0")]:
        ax.plot(trial["p"][:, 0], trial["p"][:, 1],
                label=trial["label"], color=color, alpha=0.8)
    ax.plot(trial_nom_nom["pdes"][:, 0], trial_nom_nom["pdes"][:, 1],
            "--", color="gray", alpha=0.4, label="reference")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title("Top view")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for trial, color in [(trial_nom_nom, "C2"),
                          (trial_nom_perturbed, "C3"),
                          (trial_res_perturbed, "C0")]:
        ax.plot(trial["t"], trial["err"] * 100,
                label=f"{trial['label']} (RMS={trial['rms_cm']:.2f})",
                color=color, alpha=0.8)
    ax.axvline(ERR_START_T, ls=":", color="black", alpha=0.4,
                label="metric window start")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||error|| (cm)")
    ax.set_title("Tracking error magnitude")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    out_path = os.path.join(HERE, "residual_mpc_test.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")


if __name__ == "__main__":
    main()