"""
Stage 4 — sweep of inertia perturbation.

The MPC has no explicit inertia term in its prediction (kinematic-thrust
formulation), so this sweep primarily measures how MuJoCo's rotational
dynamics interact with the body-rate inner loop, not how MPC itself
degrades. Expected to show small, possibly non-monotonic effects.

Run:
    python3 03_sweep_inertia.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stage4_src.experiment_runner import run_trial


FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def main():
    print("=" * 60)
    print("Stage 4 — inertia perturbation sweep")
    print("=" * 60)
    print()

    results = []
    for f in FACTORS:
        spec = {"type": "inertia", "value": f}
        print(f"  Inertia factor = {f:.2f} ...", end=" ", flush=True)
        result = run_trial(spec, verbose=False)
        rms_cm = result["rms_err"] * 100 if np.isfinite(result["rms_err"]) else float('inf')
        max_cm = result["max_err"] * 100 if np.isfinite(result["max_err"]) else float('inf')
        print(f"RMS = {rms_cm:.2f} cm  Max = {max_cm:.2f} cm  Stable = {result['stable']}")
        results.append({
            "factor": f,
            "rms_cm": rms_cm,
            "max_cm": max_cm,
            "stable": result["stable"],
        })

    csv_path = os.path.join(HERE, "sweep_inertia.csv")
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["factor", "rms_cm",
                                                    "max_cm", "stable"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults saved to: {csv_path}")

    factors = [r["factor"] for r in results if r["stable"]]
    rms_cm = [r["rms_cm"] for r in results if r["stable"]]
    max_cm = [r["max_cm"] for r in results if r["stable"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(factors, rms_cm, 'o-', label='RMS', color='C0', linewidth=1.5)
    ax.plot(factors, max_cm, 's--', label='Max', color='C1',
            linewidth=1.0, alpha=0.7)
    ax.axvline(1.0, ls=':', color='gray', alpha=0.5, label='nominal')
    ax.axhline(1.65, ls=':', color='C2', alpha=0.5,
                label='Stage 3 nominal RMS (1.65 cm)')
    ax.set_xlabel("Inertia factor (multiplier on nominal diagonal)")
    ax.set_ylabel("Tracking error (cm)")
    ax.set_title("Inertia perturbation — tracking error vs inertia factor")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(HERE, "sweep_inertia.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()