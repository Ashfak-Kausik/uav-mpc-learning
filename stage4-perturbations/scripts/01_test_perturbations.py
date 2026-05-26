"""
Stage 4 — sanity check that perturbations actually change behavior.

Runs 5 representative trials:
  1. No perturbation (baseline)
  2. +20% mass
  3. -20% mass
  4. +50% drag (per-axis 0.5 N*s/m)
  5. +50% inertia

Just prints metrics. If perturbations are working, the metrics should
change in the expected directions:
  - More mass: drone sags under hover thrust, max error and RMS grow.
  - Less mass: drone accelerates more aggressively, oscillations grow.
  - Drag: drone lags behind reference, RMS grows.
  - More inertia: attitude tracking is slower, transient errors grow.

Run:
    python3 01_test_perturbations.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stage4_src.experiment_runner import run_trial


def main():
    print("=" * 60)
    print("Stage 4 perturbation sanity check")
    print("=" * 60)
    print()

    specs = [
      {"type": "none",        "value": None},
      {"type": "mass",        "value": 1.20},
      {"type": "mass",        "value": 0.80},
      {"type": "drag",        "value": 0.5},
      {"type": "inertia",     "value": 1.5},
      {"type": "motor_lag",   "value": 0.05},   # 50 ms motor time constant
      {"type": "time_delay",  "value": 0.05},   # 50 ms pure delay
    ]

    for spec in specs:
        label = (f"{spec['type']}={spec['value']}"
                 if spec['value'] is not None else "no perturbation")
        print(f"\nTrial: {label}")
        result = run_trial(spec, verbose=True)
        print(f"  RMS error: {result['rms_err']*100:.2f} cm "
              f"(stable: {result['stable']})")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()