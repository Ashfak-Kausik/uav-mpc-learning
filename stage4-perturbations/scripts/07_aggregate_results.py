"""
Stage 4 — aggregate plot of all perturbation sweeps.

Loads the CSVs from each of the five (or seven with zoom) sweeps and
produces a single figure showing how MPC tracking error scales with
each perturbation type. The x-axis of each subplot is the perturbation
magnitude (units differ per type); the y-axis is RMS tracking error.

This is the headline deliverable of Stage 4 — the figure that motivates
Stage 5's learned residual.

Run:
    python3 07_aggregate_results.py
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Mapping from sweep CSV filename to plot configuration.
# Each entry: (csv_path, x_column, x_label, title, optional_zoom_csv)
SWEEPS = [
    {
        "csv": "sweep_mass.csv",
        "x_col": "factor",
        "x_label": "Mass factor (multiplier on 1.325 kg)",
        "title": "Mass",
        "nominal_x": 1.0,
        "zoom_csv": None,
    },
    {
        "csv": "sweep_inertia.csv",
        "x_col": "factor",
        "x_label": "Inertia factor",
        "title": "Inertia",
        "nominal_x": 1.0,
        "zoom_csv": None,
    },
    {
        "csv": "sweep_drag.csv",
        "x_col": "coefficient",
        "x_label": "Drag coefficient (N s/m, per axis)",
        "title": "Drag",
        "nominal_x": 0.0,
        "zoom_csv": None,
    },
    {
        "csv": "sweep_motor_lag.csv",
        "x_col": "tau_ms",
        "x_label": "Motor lag tau (ms)",
        "title": "Motor lag",
        "nominal_x": 0.0,
        "zoom_csv": "sweep_motor_lag_zoom.csv",
    },
    {
        "csv": "sweep_time_delay.csv",
        "x_col": "delay_ms",
        "x_label": "Time delay (ms)",
        "title": "Time delay",
        "nominal_x": 0.0,
        "zoom_csv": "sweep_time_delay_zoom.csv",
    },
]

NOMINAL_RMS = 1.65   # Stage 3 baseline


def load_csv(filename):
    """Load a sweep CSV. Returns list of rows as dicts."""
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric columns
            for k, v in row.items():
                if k == "stable":
                    row[k] = (v.lower() == "true")
                else:
                    try:
                        row[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    return rows


def plot_sweep_subplot(ax, sweep_cfg):
    """Plot one sweep on the given axes."""
    rows = load_csv(sweep_cfg["csv"])
    if rows is None:
        ax.text(0.5, 0.5, f"Missing: {sweep_cfg['csv']}",
                ha="center", va="center", transform=ax.transAxes,
                color="red")
        ax.set_title(sweep_cfg["title"])
        return

    # Merge with zoom data if present
    if sweep_cfg["zoom_csv"]:
        zoom_rows = load_csv(sweep_cfg["zoom_csv"])
        if zoom_rows is not None:
            # Dedup by x value (zoom takes precedence)
            x_col = sweep_cfg["x_col"]
            zoom_x = set(r[x_col] for r in zoom_rows)
            main_rows = [r for r in rows if r[x_col] not in zoom_x]
            rows = sorted(main_rows + zoom_rows, key=lambda r: r[x_col])

    x_col = sweep_cfg["x_col"]
    x_stable = [r[x_col] for r in rows if r["stable"]]
    rms_stable = [r["rms_cm"] for r in rows if r["stable"]]
    x_crashed = [r[x_col] for r in rows if not r["stable"]]

    ax.plot(x_stable, rms_stable, 'o-', color='C0',
            linewidth=1.5, markersize=5, label='RMS (stable)')
    if x_crashed:
        y_crash = (max(rms_stable) if rms_stable else NOMINAL_RMS) * 1.15
        ax.scatter(x_crashed, [y_crash] * len(x_crashed),
                   marker='x', color='red', s=70, linewidth=2,
                   label=f"crash ({len(x_crashed)})")

    ax.axhline(NOMINAL_RMS, ls=':', color='C2', alpha=0.6, linewidth=1)
    ax.axvline(sweep_cfg["nominal_x"], ls=':', color='gray',
               alpha=0.4, linewidth=1)
    ax.set_xlabel(sweep_cfg["x_label"], fontsize=9)
    ax.set_ylabel("RMS error (cm)", fontsize=9)
    ax.set_title(sweep_cfg["title"], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)


def main():
    print("=" * 60)
    print("Stage 4 — aggregate perturbation results")
    print("=" * 60)
    print()

    # Print summary table to stdout
    print(f"  Stage 3 nominal (no perturbation): RMS = {NOMINAL_RMS} cm")
    print()
    print(f"  Per perturbation type, worst stable RMS and stability:")
    print()
    print(f"  {'Type':<14} {'Worst stable RMS':<20} {'Crashes':<10}")
    print(f"  {'-' * 14} {'-' * 20} {'-' * 10}")
    for cfg in SWEEPS:
        rows = load_csv(cfg["csv"])
        if rows is None:
            continue
        if cfg["zoom_csv"]:
            zoom_rows = load_csv(cfg["zoom_csv"])
            if zoom_rows is not None:
                x_col = cfg["x_col"]
                zoom_x = set(r[x_col] for r in zoom_rows)
                main_rows = [r for r in rows if r[x_col] not in zoom_x]
                rows = sorted(main_rows + zoom_rows, key=lambda r: r[x_col])
        stable_rms = [r["rms_cm"] for r in rows if r["stable"]]
        n_crashed = sum(1 for r in rows if not r["stable"])
        worst = max(stable_rms) if stable_rms else float("nan")
        print(f"  {cfg['title']:<14} {worst:>6.2f} cm{'':<11} "
              f"{n_crashed:>3}")

    # Build the figure: 5 subplots in a 2x3 grid (last cell empty)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for i, cfg in enumerate(SWEEPS):
        plot_sweep_subplot(axes_flat[i], cfg)

    # Hide the unused 6th subplot
    axes_flat[-1].axis("off")

    # Add a global subtitle
    fig.suptitle(
        "Stage 4 — MPC tracking error vs perturbation magnitude\n"
        f"(Nominal MPC tracking on 1.5 m circle: RMS = {NOMINAL_RMS} cm)",
        fontsize=13, fontweight='bold', y=1.00,
    )

    out_path = os.path.join(HERE, "aggregate_perturbations.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nAggregate figure saved to: {out_path}")

    # Also print quick observations to stdout, for paper-writing later
    print()
    print("=" * 60)
    print("Quick observations (from the data):")
    print("=" * 60)
    print()
    print("1. Mass: V-shaped curve, symmetric. ~6x degradation at ±40%.")
    print("2. Inertia: nearly flat except at extremes; reflects the")
    print("   kinematic-thrust MPC's insensitivity to inertia.")
    print("3. Drag: small drag (b ~ 0.1) acts as passive damping and")
    print("   slightly improves tracking; larger drag degrades it.")
    print("4. Motor lag: binary effect — no impact below ~30 ms, then")
    print("   sharp transition, then crash. Driven by controller-timescale.")
    print("5. Time delay: sharpest cliff — stable through ~10 ms, then")
    print("   immediate crash. Classical delay-instability behavior.")


if __name__ == "__main__":
    main()