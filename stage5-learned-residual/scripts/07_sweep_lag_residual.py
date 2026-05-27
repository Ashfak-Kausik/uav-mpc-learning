"""
Stage 5 evaluation — motor lag perturbation sweep with feedforward residual.

For each motor lag tau in {0, 10, 20, 30, 40} ms:
  1. Nominal MPC + nominal plant (baseline, run once)
  2. Nominal MPC + motor lag perturbation
  3. Feedforward Residual + motor lag perturbation

Lag values stay below the 45-50 ms cliff observed in Stage 4 to avoid
trial-wasting crashes.

Run:
    python3 07_sweep_lag_residual.py
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
if STAGE3_ROOT not in sys.path:
    sys.path.insert(0, STAGE3_ROOT)

from stage3_src.mpc_controller import MPCController

from stage5_src.casadi_residual import (
    build_casadi_residual, load_pytorch_model,
)
from stage5_src.feedforward_residual_controller import (
    FeedforwardResidualController,
)
from stage5_src.feedforward_experiment_runner import run_trial


MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "models"))
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "residual_best.pt")

# Lag values in seconds. Stay below the 0.045s cliff Stage 4 found.
LAG_TAUS = [0.0, 0.020, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049]

OUT_CSV = os.path.join(HERE, "sweep_lag_residual.csv")
OUT_PNG = os.path.join(HERE, "sweep_lag_residual.png")


def main():
    print("=" * 60)
    print("Stage 5 — motor lag sweep with feedforward residual")
    print("=" * 60)

    print(f"\nLoading residual model from {BEST_MODEL_PATH}")
    pytorch_model = load_pytorch_model(BEST_MODEL_PATH)
    ca_residual = build_casadi_residual(pytorch_model)

    rows = []
    t_total_start = time.time()

    # Baseline
    print("\nBaseline (nominal MPC + nominal plant)...")
    baseline_ctrl = MPCController()
    baseline_result = run_trial(baseline_ctrl, {"type": "none"})
    print(f"  RMS = {baseline_result['rms_cm']:.2f} cm")
    baseline_rms = baseline_result["rms_cm"]
    rows.append({
        "config": "baseline", "lag_tau_s": 0.0,
        "controller": "nominal", "perturbed": False,
        **baseline_result,
    })

    for tau in LAG_TAUS:
        tau_ms = int(tau * 1000)
        # Nominal MPC under motor lag
        print(f"\n[nominal MPC, lag tau={tau_ms} ms]")
        ctrl = MPCController()
        spec = {"type": "motor_lag", "value": tau}
        res = run_trial(ctrl, spec)
        print(f"  RMS = {res['rms_cm']:.2f} cm  crashed={res['crashed']}")
        rows.append({
            "config": "nominal_perturbed", "lag_tau_s": tau,
            "controller": "nominal", "perturbed": True, **res,
        })

        # Feedforward residual under motor lag
        print(f"\n[ff residual, lag tau={tau_ms} ms]")
        ctrl = FeedforwardResidualController(
            residual_fn=ca_residual, sim_dt=0.01,
            thrust_correction_gain=1.0,
        )
        res = run_trial(ctrl, spec)
        print(f"  RMS = {res['rms_cm']:.2f} cm  crashed={res['crashed']}  "
              f"thrust_corr_mean={res['thrust_correction_mean']:+.3f} N")
        rows.append({
            "config": "ff_residual_perturbed", "lag_tau_s": tau,
            "controller": "ff_residual", "perturbed": True, **res,
        })

    elapsed = time.time() - t_total_start
    print(f"\nTotal sweep time: {elapsed/60:.1f} min")

    # CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV saved to: {OUT_CSV}")

    # Aggregate
    nom_rms = []
    ff_rms = []
    for tau in LAG_TAUS:
        n = next(r for r in rows
                 if r["config"] == "nominal_perturbed"
                 and r["lag_tau_s"] == tau)
        ff = next(r for r in rows
                  if r["config"] == "ff_residual_perturbed"
                  and r["lag_tau_s"] == tau)
        nom_rms.append(n["rms_cm"])
        ff_rms.append(ff["rms_cm"])
    nom_rms = np.array(nom_rms)
    ff_rms = np.array(ff_rms)
    taus_ms = [t * 1000 for t in LAG_TAUS]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(taus_ms, nom_rms, "o-", color="C3", label="Nominal MPC")
    ax.plot(taus_ms, ff_rms, "o-", color="C0",
            label="Feedforward Residual MPC")
    ax.axhline(baseline_rms, ls="--", color="C2", alpha=0.6,
                label=f"Unperturbed baseline ({baseline_rms:.2f} cm)")
    ax.set_xlabel("Motor lag time constant (ms)")
    ax.set_ylabel("Tracking RMS error (cm)")
    ax.set_title("Tracking error vs motor lag")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    pct = []
    for i, tau in enumerate(LAG_TAUS):
        gap = nom_rms[i] - baseline_rms
        recovered = nom_rms[i] - ff_rms[i]
        if abs(gap) > 0.01:
            pct.append(recovered / gap * 100)
        else:
            pct.append(float("nan"))
    pct = np.array(pct)
    bar_colors = []
    for p in pct:
        if np.isnan(p):
            bar_colors.append("C7")
        elif p >= 50:
            bar_colors.append("C0")
        elif p >= 0:
            bar_colors.append("C7")
        else:
            bar_colors.append("C3")
    ax.bar(taus_ms, pct, width=4, color=bar_colors)
    ax.axhline(100, ls=":", color="C2", alpha=0.6)
    ax.axhline(0, ls="-", color="black", alpha=0.4)
    ax.set_xlabel("Motor lag time constant (ms)")
    ax.set_ylabel("Gap closure (%)")
    ax.set_title("Residual recovery: gap closure")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(-30, 130)
    for i, tau in enumerate(LAG_TAUS):
        if not np.isnan(pct[i]):
            ax.text(taus_ms[i], pct[i] + 3, f"{pct[i]:.0f}%",
                    ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Figure saved to: {OUT_PNG}")

    print("\nSUMMARY")
    print(f"  {'lag(ms)':>7} {'nominal (cm)':>14} {'ff resid (cm)':>14} "
          f"{'closure':>10}")
    for i, tau in enumerate(LAG_TAUS):
        c = "n/a" if np.isnan(pct[i]) else f"{pct[i]:.0f}%"
        print(f"  {taus_ms[i]:>7.0f} {nom_rms[i]:>14.2f} "
              f"{ff_rms[i]:>14.2f} {c:>10}")


if __name__ == "__main__":
    main()