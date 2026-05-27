"""
Stage 5 evaluation — mass perturbation sweep with feedforward residual.

For each mass factor in {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4},
runs three trials per seed (3 seeds = 9 trials per mass factor):
  1. Nominal MPC + nominal plant (control baseline, runs once total)
  2. Nominal MPC + perturbed plant (Stage 4 baseline)
  3. Feedforward Residual + perturbed plant (Stage 5 result)

Reports tracking RMS at each mass factor and saves to CSV.

Note: nominal+nominal trial is run once (seed=0) and reused as the
baseline reference since it doesn't depend on the mass factor.

Run:
    python3 05_sweep_mass_residual.py
"""

import os
import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

STAGE3_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage3-mpc-baseline"))
STAGE4_ROOT = os.path.abspath(os.path.join(HERE, "..", "..",
                                            "stage4-perturbations"))
for p in [STAGE3_ROOT, STAGE4_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from stage3_src.mpc_controller import MPCController
from stage4_src.perturbations import apply_mass_perturbation

from stage5_src.casadi_residual import (
    build_casadi_residual, load_pytorch_model,
)
from stage5_src.feedforward_residual_controller import (
    FeedforwardResidualController,
)
from stage5_src.feedforward_experiment_runner import run_trial


MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "models"))
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "residual_best.pt")

# Match Stage 4's mass sweep configuration
MASS_FACTORS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
SEEDS = [0, 1, 2]

OUT_CSV = os.path.join(HERE, "sweep_mass_residual.csv")
OUT_PNG = os.path.join(HERE, "sweep_mass_residual.png")


def main():
    print("=" * 60)
    print("Stage 5 — mass perturbation sweep with feedforward residual")
    print("=" * 60)

    # Load model once
    print(f"\nLoading residual model from {BEST_MODEL_PATH}")
    pytorch_model = load_pytorch_model(BEST_MODEL_PATH)
    ca_residual = build_casadi_residual(pytorch_model)

    rows = []
    t_total_start = time.time()

    # First: one baseline trial (nominal MPC + nominal plant)
    print("\nBaseline (nominal MPC + nominal plant)...")
    baseline_ctrl = MPCController()
    baseline_result = run_trial(baseline_ctrl, apply_perturbation_fn=None,
                                 label="baseline_nominal_nominal")
    print(f"  RMS = {baseline_result['rms_cm']:.2f} cm")
    baseline_rms = baseline_result["rms_cm"]

    rows.append({
        "config": "baseline",
        "mass_factor": 1.0,
        "seed": -1,
        "controller": "nominal",
        "perturbed": False,
        **baseline_result,
    })

    # For each mass factor and seed, run nominal-perturbed and ff-perturbed
    for mass_factor in MASS_FACTORS:
        for seed in SEEDS:
            np.random.seed(seed)

            # Trial type A: nominal MPC under this mass perturbation
            label_nom = f"nominal_mass{mass_factor}_seed{seed}"
            print(f"\n[{label_nom}]")
            ctrl = MPCController()  # fresh controller (no warm-start contamination)
            perturb_fn = lambda m, mf=mass_factor: apply_mass_perturbation(m, mf)
            res = run_trial(ctrl, apply_perturbation_fn=perturb_fn,
                             label=label_nom)
            print(f"  RMS = {res['rms_cm']:.2f} cm "
                  f"crashed={res['crashed']}")
            rows.append({
                "config": "nominal_perturbed",
                "mass_factor": mass_factor,
                "seed": seed,
                "controller": "nominal",
                "perturbed": True,
                **res,
            })

            # Trial type B: feedforward residual under this mass perturbation
            label_ff = f"ff_residual_mass{mass_factor}_seed{seed}"
            print(f"\n[{label_ff}]")
            ctrl = FeedforwardResidualController(
                residual_fn=ca_residual,
                sim_dt=0.01,
                thrust_correction_gain=1.0,
            )
            res = run_trial(ctrl, apply_perturbation_fn=perturb_fn,
                             label=label_ff)
            print(f"  RMS = {res['rms_cm']:.2f} cm "
                  f"crashed={res['crashed']}  "
                  f"thrust_corr_mean={res['thrust_correction_mean']:+.3f} N")
            rows.append({
                "config": "ff_residual_perturbed",
                "mass_factor": mass_factor,
                "seed": seed,
                "controller": "ff_residual",
                "perturbed": True,
                **res,
            })

    elapsed_total = time.time() - t_total_start
    print(f"\nTotal sweep time: {elapsed_total/60:.1f} min")

    # Save CSV
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV saved to: {OUT_CSV}")

    # ---- Aggregate and plot ----
    def aggregate(config_name):
        out = {}
        for mf in MASS_FACTORS:
            vals = [r["rms_cm"] for r in rows
                    if r["config"] == config_name
                    and r["mass_factor"] == mf
                    and not r["crashed"]
                    and not np.isnan(r["rms_cm"])]
            if vals:
                out[mf] = (np.mean(vals), np.std(vals), len(vals))
            else:
                out[mf] = (np.nan, np.nan, 0)
        return out

    agg_nominal = aggregate("nominal_perturbed")
    agg_ff = aggregate("ff_residual_perturbed")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: degradation curves
    ax = axes[0]
    mfs = np.array(MASS_FACTORS)
    nom_mean = np.array([agg_nominal[mf][0] for mf in MASS_FACTORS])
    nom_std = np.array([agg_nominal[mf][1] for mf in MASS_FACTORS])
    ff_mean = np.array([agg_ff[mf][0] for mf in MASS_FACTORS])
    ff_std = np.array([agg_ff[mf][1] for mf in MASS_FACTORS])

    ax.errorbar(mfs, nom_mean, yerr=nom_std,
                fmt="o-", color="C3", label="Nominal MPC", capsize=4)
    ax.errorbar(mfs, ff_mean, yerr=ff_std,
                fmt="o-", color="C0", label="Feedforward Residual MPC",
                capsize=4)
    ax.axhline(baseline_rms, ls="--", color="C2", alpha=0.6,
                label=f"Unperturbed baseline ({baseline_rms:.2f} cm)")
    ax.axvline(1.0, ls=":", color="gray", alpha=0.4)
    ax.set_xlabel("Mass factor (× nominal)")
    ax.set_ylabel("Tracking RMS error (cm)")
    ax.set_title("Tracking error vs mass perturbation")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Right panel: gap closure percentage
    ax = axes[1]
    pct = np.zeros_like(mfs)
    for i, mf in enumerate(MASS_FACTORS):
        gap = nom_mean[i] - baseline_rms
        recovered = nom_mean[i] - ff_mean[i]
        if gap > 1e-3:
            pct[i] = recovered / gap * 100
        else:
            pct[i] = np.nan
    ax.bar(mfs, pct, width=0.06,
           color=["C0" if p >= 50 else ("C7" if p >= 0 else "C3")
                  for p in pct])
    ax.axhline(100, ls=":", color="C2", alpha=0.6)
    ax.axhline(0, ls="-", color="black", alpha=0.4)
    ax.set_xlabel("Mass factor (× nominal)")
    ax.set_ylabel("Gap closure (%)")
    ax.set_title("Residual recovery: (nominal − residual) / (nominal − baseline)")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(-20, 130)
    for i, mf in enumerate(MASS_FACTORS):
        if not np.isnan(pct[i]):
            ax.text(mf, pct[i] + 3, f"{pct[i]:.0f}%",
                    ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Figure saved to: {OUT_PNG}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'mass':>5} {'nominal (cm)':>14} {'ff resid (cm)':>14} "
          f"{'gap closure':>12}")
    print(f"  {'-'*5} {'-'*14} {'-'*14} {'-'*12}")
    for i, mf in enumerate(MASS_FACTORS):
        nm = nom_mean[i]
        fm = ff_mean[i]
        print(f"  {mf:>5.2f} {nm:>14.2f} {fm:>14.2f} "
              f"{pct[i]:>11.0f}%")


if __name__ == "__main__":
    main()