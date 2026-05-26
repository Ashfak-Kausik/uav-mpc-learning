"""
Stage 4 — sweep of motor lag (first-order time constant).

Each of the four motor commands is passed through a first-order filter
with time constant tau before being applied to MuJoCo. The MPC's model
assumes zero lag.

Run:
    python3 05_sweep_motor_lag.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stage4_src.experiment_runner import run_trial


TAU_VALUES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]


def main():
    print("=" * 60)
    print("Stage 4 — motor lag sweep")
    print("=" * 60)
    print()

    results = []
    for tau in TAU_VALUES:
        spec = {"type": "motor_lag", "value": tau}
        print(f"  Motor lag tau = {tau*1000:.0f} ms ...", end=" ", flush=True)
        result = run_trial(spec, verbose=False)
        rms_cm = result["rms_err"] * 100 if np.isfinite(result["rms_err"]) else float('inf')
        max_cm = result["max_err"] * 100 if np.isfinite(result["max_err"]) else float('inf')
        print(f"RMS = {rms_cm:.2f} cm  Max = {max_cm:.2f} cm  Stable = {result['stable']}")
        results.append({
            "tau_ms": tau * 1000,
            "rms_cm": rms_cm,
            "max_cm": max_cm,
            "stable": result["stable"],
        })

    csv_path = os.path.join(HERE, "sweep_motor_lag.csv")
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["tau_ms", "rms_cm",
                                                    "max_cm", "stable"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults saved to: {csv_path}")

    taus = [r["tau_ms"] for r in results if r["stable"]]
    rms_cm = [r["rms_cm"] for r in results if r["stable"]]
    max_cm = [r["max_cm"] for r in results if r["stable"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, rms_cm, 'o-', label='RMS', color='C0', linewidth=1.5)
    ax.plot(taus, max_cm, 's--', label='Max', color='C1',
            linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, ls=':', color='gray', alpha=0.5, label='nominal (tau=0)')
    ax.axhline(1.65, ls=':', color='C2', alpha=0.5,
                label='Stage 3 nominal RMS (1.65 cm)')
    ax.set_xlabel("Motor lag time constant (ms)")
    ax.set_ylabel("Tracking error (cm)")
    ax.set_title("Motor lag perturbation — tracking error vs tau")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(HERE, "sweep_motor_lag.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()