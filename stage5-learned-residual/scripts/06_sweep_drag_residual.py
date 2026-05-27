"""
Stage 5 evaluation — drag perturbation sweep with feedforward residual.

For each drag coefficient in {0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0}:
  1. Nominal MPC + nominal plant (baseline, run once)
  2. Nominal MPC + drag perturbation
  3. Feedforward Residual + drag perturbation

Single seed per level (perturbation is deterministic).

Run:
    python3 06_sweep_drag_residual.py
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

DRAG_COEFFS = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

OUT_CSV = os.path.join(HERE, "sweep_drag_residual.csv")
OUT_PNG = os.path.join(HERE, "sweep_drag_residual.png")


def main():
    print("=" * 60)
    print("Stage 5 — drag perturbation sweep with feedforward residual")
    print("=" * 60)

    print(f"\nLoading residual model from {BEST_MODEL_PATH}")
    pytorch_model = load_pytorch_model(BEST_MODEL_PATH)
    ca_residual = build_casadi_residual(pytorch_model)

    rows = []
    t_total_start = time.time()

    # Baseline (no perturbation, nominal MPC)
    print("\nBaseline (nominal MPC + nominal plant)...")
    baseline_ctrl = MPCController()
    baseline_result = run_trial(baseline_ctrl, {"type": "none"})
    print(f"  RMS = {baseline_result['rms_cm']:.2f} cm")
    baseline_rms = baseline_result["rms_cm"]
    rows.append({
        "config": "baseline", "drag_coeff": 0.0,
        "controller": "nominal", "perturbed": False,
        **baseline_result,
    })

    for b in DRAG_COEFFS:
        # Nominal MPC under drag
        print(f"\n[nominal MPC, drag b={b}]")
        ctrl = MPCController()
        spec = {"type": "drag", "value": b}
        res = run_trial(ctrl, spec)
        print(f"  RMS = {res['rms_cm']:.2f} cm  crashed={res['crashed']}")
        rows.append({
            "config": "nominal_perturbed", "drag_coeff": b,
            "controller": "nominal", "perturbed": True, **res,
        })

        # Feedforward residual under drag
        print(f"\n[ff residual, drag b={b}]")
        ctrl = FeedforwardResidualController(
            residual_fn=ca_residual, sim_dt=0.01,
            thrust_correction_gain=1.0,
        )
        res = run_trial(ctrl, spec)
        print(f"  RMS = {res['rms_cm']:.2f} cm  crashed={res['crashed']}  "
              f"thrust_corr_mean={res['thrust_correction_mean']:+.3f} N")
        rows.append({
            "config": "ff_residual_perturbed", "drag_coeff": b,
            "controller": "ff_residual", "perturbed": True, **res,
        })

    elapsed = time.time() - t_total_start
    print(f"\nTotal sweep time: {elapsed/60:.1f} min")

    # Save CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV saved to: {OUT_CSV}")

    # Aggregate for plotting
    nom_rms = []
    ff_rms = []
    for b in DRAG_COEFFS:
        n_row = next(r for r in rows
                     if r["config"] == "nominal_perturbed"
                     and r["drag_coeff"] == b)
        f_row = next(r for r in rows
                     if r["config"] == "ff_residual_perturbed"
                     and r["drag_coeff"] == b)
        nom_rms.append(n_row["rms_cm"])
        ff_rms.append(f_row["rms_cm"])
    nom_rms = np.array(nom_rms)
    ff_rms = np.array(ff_rms)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(DRAG_COEFFS, nom_rms, "o-", color="C3", label="Nominal MPC")
    ax.plot(DRAG_COEFFS, ff_rms, "o-", color="C0",
            label="Feedforward Residual MPC")
    ax.axhline(baseline_rms, ls="--", color="C2", alpha=0.6,
                label=f"Unperturbed baseline ({baseline_rms:.2f} cm)")
    ax.set_xlabel("Drag coefficient b")
    ax.set_ylabel("Tracking RMS error (cm)")
    ax.set_title("Tracking error vs drag perturbation")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    pct = []
    for i, b in enumerate(DRAG_COEFFS):
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
    ax.bar(DRAG_COEFFS, pct, width=0.08, color=bar_colors)
    ax.axhline(100, ls=":", color="C2", alpha=0.6)
    ax.axhline(0, ls="-", color="black", alpha=0.4)
    ax.set_xlabel("Drag coefficient b")
    ax.set_ylabel("Gap closure (%)")
    ax.set_title("Residual recovery: gap closure")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(-30, 130)
    for i, b in enumerate(DRAG_COEFFS):
        if not np.isnan(pct[i]):
            ax.text(b, pct[i] + 3, f"{pct[i]:.0f}%",
                    ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Figure saved to: {OUT_PNG}")

    print("\nSUMMARY")
    print(f"  {'drag':>5} {'nominal (cm)':>14} {'ff resid (cm)':>14} "
          f"{'closure':>10}")
    for i, b in enumerate(DRAG_COEFFS):
        c = "n/a" if np.isnan(pct[i]) else f"{pct[i]:.0f}%"
        print(f"  {b:>5.2f} {nom_rms[i]:>14.2f} {ff_rms[i]:>14.2f} "
              f"{c:>10}")


if __name__ == "__main__":
    main()