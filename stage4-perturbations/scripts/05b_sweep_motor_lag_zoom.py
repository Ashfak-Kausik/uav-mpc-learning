"""
Stage 4 — fine-grained motor lag sweep around the transition.

The main motor-lag sweep showed a sharp cliff between tau = 30 ms (no
effect) and tau = 50 ms (12 cm RMS, then crash at 70 ms). This zoom
sweep fills in the cliff with finer resolution to characterize the
exact transition.

Run:
    python3 05b_sweep_motor_lag_zoom.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stage4_src.experiment_runner import run_trial


TAU_VALUES = [0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065]


def main():
    print("=" * 60)
    print("Stage 4 — motor lag zoom sweep (cliff region)")
    print("=" * 60)
    print()

    results = []
    for tau in TAU_VALUES:
        spec = {"type": "motor_lag", "value": tau}
        print(f"  Motor lag tau = {tau*1000:.0f} ms ...", end=" ", flush=True)
        result = run_trial(spec, verbose=False)
        rms_cm = result["rms_err"] * 100 if np.isfinite(result["rms_err"]) else float('inf')
        max_cm = result["max_err"] * 100 if np.isfinite(result["max_err"]) else float('inf')
        status = "stable" if result["stable"] else "CRASHED"
        if result["stable"]:
            print(f"RMS = {rms_cm:.2f} cm  Max = {max_cm:.2f} cm  {status}")
        else:
            print(f"{status}")
        results.append({
            "tau_ms": tau * 1000,
            "rms_cm": rms_cm,
            "max_cm": max_cm,
            "stable": result["stable"],
        })

    csv_path = os.path.join(HERE, "sweep_motor_lag_zoom.csv")
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
    taus_crashed = [r["tau_ms"] for r in results if not r["stable"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, rms_cm, 'o-', label='RMS (stable)',
            color='C0', linewidth=1.5)
    ax.plot(taus, max_cm, 's--', label='Max (stable)',
            color='C1', linewidth=1.0, alpha=0.7)
    if taus_crashed:
        y_crash = (max(rms_cm) if rms_cm else 1.65) * 1.3
        ax.scatter(taus_crashed,
                   [y_crash] * len(taus_crashed),
                   marker='x', color='red', s=80,
                   label=f"CRASHED ({len(taus_crashed)} trials)")
    ax.axhline(1.65, ls=':', color='C2', alpha=0.5,
                label='Stage 3 nominal RMS (1.65 cm)')
    ax.set_xlabel("Motor lag time constant (ms)")
    ax.set_ylabel("Tracking error (cm)")
    ax.set_title("Motor lag — cliff zoom (30-65 ms)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(HERE, "sweep_motor_lag_zoom.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()