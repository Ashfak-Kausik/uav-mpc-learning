"""
Stage 5 — collect rollout data under all perturbation configurations.

Runs the Stage 3 nominal MPC on the circle trajectory across a planned
set of perturbations and seeds. Each rollout is saved as a .npz file
and all rollouts are also stacked into a single combined.npz at the end.

Total: ~63 rollouts at ~6s each = ~6 minutes runtime.

Run:
    python3 01_collect_data.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stage5_src.data_collector import collect_rollout, save_rollout


# Where to save data
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Number of seeds per perturbation configuration
N_SEEDS = 3

# Perturbation configurations to collect
SWEEPS = {
    "mass": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    "drag": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    "motor_lag": [0.0, 0.010, 0.020, 0.030, 0.040],
    "time_delay": [0.0, 0.005, 0.010, 0.014],
}


def build_combined_dataset(rollout_paths, output_path):
    """Stack all rollouts into a single combined .npz for training."""
    print(f"\nBuilding combined dataset from {len(rollout_paths)} rollouts...")

    all_state, all_control, all_state_next, all_nominal = [], [], [], []
    all_ptype, all_pvalue = [], []

    for path in rollout_paths:
        d = np.load(path, allow_pickle=True)
        if bool(d["crashed"]):
            print(f"  Skipping crashed rollout: {os.path.basename(path)}")
            continue
        all_state.append(d["state"])
        all_control.append(d["control"])
        all_state_next.append(d["state_next"])
        all_nominal.append(d["state_next_nominal"])

        # Per-step labels (same value repeated for all steps in the rollout)
        n = d["state"].shape[0]
        all_ptype.extend([str(d["perturbation_type"])] * n)
        all_pvalue.extend([float(d["perturbation_value"])] * n)

    state       = np.concatenate(all_state, axis=0)
    control     = np.concatenate(all_control, axis=0)
    state_next  = np.concatenate(all_state_next, axis=0)
    nominal     = np.concatenate(all_nominal, axis=0)
    ptype_arr   = np.array(all_ptype)
    pvalue_arr  = np.array(all_pvalue)

    residual = state_next - nominal

    print(f"  Combined dataset:")
    print(f"    Transitions:        {state.shape[0]}")
    print(f"    State dim:          {state.shape[1]}")
    print(f"    Control dim:        {control.shape[1]}")
    print(f"    Residual magnitude per axis (RMS):")
    for i, name in enumerate(["px", "py", "pz", "vx", "vy", "vz",
                               "roll", "pitch", "yaw"]):
        rms = float(np.sqrt(np.mean(residual[:, i] ** 2)))
        print(f"      {name:6s}: {rms:.6f}")

    np.savez(output_path,
             state=state, control=control,
             state_next=state_next,
             state_next_nominal=nominal,
             residual=residual,
             perturbation_type=ptype_arr,
             perturbation_value=pvalue_arr)
    print(f"  Saved to: {output_path}")


def main():
    print("=" * 60)
    print("Stage 5 — data collection")
    print("=" * 60)

    rollout_paths = []
    total_rollouts = sum(len(values) for values in SWEEPS.values()) * N_SEEDS
    rollout_idx = 0
    t_start = time.time()

    for ptype, values in SWEEPS.items():
        for value in values:
            for seed in range(N_SEEDS):
                rollout_idx += 1
                label = f"{ptype}={value}, seed={seed}"
                print(f"\n[{rollout_idx}/{total_rollouts}] Collecting {label}")

                spec = {"type": ptype, "value": value}
                rollout = collect_rollout(spec, seed=seed)
                path = save_rollout(rollout, DATA_DIR)
                rollout_paths.append(path)

                if rollout["crashed"]:
                    print(f"  CRASHED at step {rollout['n_steps']}")
                else:
                    print(f"  {rollout['n_steps']} transitions, "
                          f"saved to {os.path.basename(path)}")

    # Build combined dataset
    combined_path = os.path.join(DATA_DIR, "combined.npz")
    build_combined_dataset(rollout_paths, combined_path)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f} s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()