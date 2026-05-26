"""
Stage 4 — fine-grained time delay sweep around the transition.

The main time-delay sweep showed a sharp transition between 10 ms
(stable, no effect) and 15 ms (catastrophic crash). This zoom sweep
fills in the cliff at 11-14 ms to pin down the exact transition.

Run:
    python3 06b_sweep_time_delay_zoom.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stage4_src.experiment_runner import run_trial


DELAY_VALUES = [0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017]


def main():
    print("=" * 60)
    print("Stage 4 — time delay zoom sweep (cliff region)")
    print("=" * 60)
    print()

    results = []
    for d in DELAY_VALUES:
        spec = {"type": "time_delay", "value": d}
        print(f"  Time delay = {d*1000:.0f} ms ...", end=" ", flush=True)
        result = run_trial(spec, verbose=False)
        rms_cm = result["rms_err"] * 100 if np.isfinite(result["rms_err"]) else float('inf')
        max_cm = result["max_err"] * 100 if np.isfinite(result["max_err"]) else float('inf')
        status = "stable" if result["stable"] else "CRASHED"
        if result["stable"]:
            print(f"RMS = {rms_cm:.2f} cm  Max = {max_cm:.2f} cm  {status}")
        else:
            print(f"{status}")
        results.append({
            "delay_ms": d * 1000,
            "rms_cm": rms_cm,
            "max_cm": max_cm,
            "stable": result["stable"],
        })

    csv_path = os.path.join(HERE, "sweep_time_delay_zoom.csv")
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["delay_ms", "rms_cm",
                                                    "max_cm", "stable"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults saved to: {csv_path}")

    delays = [r["delay_ms"] for r in results if r["stable"]]
    rms_cm = [r["rms_cm"] for r in results if r["stable"]]
    max_cm = [r["max_cm"] for r in results if r["stable"]]
    delays_crashed = [r["delay_ms"] for r in results if not r["stable"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delays, rms_cm, 'o-', label='RMS (stable)',
            color='C0', linewidth=1.5)
    ax.plot(delays, max_cm, 's--', label='Max (stable)',
            color='C1', linewidth=1.0, alpha=0.7)
    if delays_crashed:
        y_crash = (max(rms_cm) if rms_cm else 1.65) * 1.3
        ax.scatter(delays_crashed,
                   [y_crash] * len(delays_crashed),
                   marker='x', color='red', s=80,
                   label=f"CRASHED ({len(delays_crashed)} trials)")
    ax.axhline(1.65, ls=':', color='C2', alpha=0.5,
                label='Stage 3 nominal RMS (1.65 cm)')
    ax.set_xlabel("Time delay (ms)")
    ax.set_ylabel("Tracking error (cm)")
    ax.set_title("Time delay — cliff zoom (10-17 ms)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(HERE, "sweep_time_delay_zoom.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()