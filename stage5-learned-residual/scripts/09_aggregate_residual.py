"""
Stage 5 aggregate figure — all four perturbation sweeps in one view.

Reads the per-sweep CSVs and produces a 2x2 panel figure showing:
  - Top-left:  mass perturbation     (mass factor 0.7-1.4)
  - Top-right: drag perturbation     (drag coefficient 0-2.0)
  - Bottom-left:  motor lag          (tau 0-49 ms, includes transition zone)
  - Bottom-right: time delay         (delay 0-14 ms, all below the cliff)

For each panel:
  - Red curve: Nominal MPC under perturbation
  - Blue curve: Feedforward residual MPC under perturbation
  - Green dashed: Unperturbed baseline reference

The figure tells the headline Stage 5 story: residual works when
architecture matches (mass), is neutral when mismatched but stable
(drag, sub-cliff lag), and destabilizes near cliffs (transition-zone
motor lag).

Run:
    python3 09_aggregate_residual.py
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Input CSVs
CSV_MASS  = os.path.join(HERE, "sweep_mass_residual.csv")
CSV_DRAG  = os.path.join(HERE, "sweep_drag_residual.csv")
CSV_LAG   = os.path.join(HERE, "sweep_lag_residual.csv")
CSV_DELAY = os.path.join(HERE, "sweep_delay_residual.csv")

OUT_PNG = os.path.join(HERE, "stage5_aggregate.png")


def load_csv(path):
    """Return list of dict rows from CSV."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def parse_float(s):
    """Robust float parser; returns nan on empty/None/'nan'."""
    if s is None or s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_bool(s):
    return str(s).strip().lower() in ("true", "1", "yes")


def aggregate_sweep(csv_path, x_key, scale_x=1.0):
    """
    Aggregate a sweep CSV into nominal/residual curves.

    Args:
        csv_path : path to sweep CSV
        x_key    : the column name holding the perturbation value
        scale_x  : multiplier for x-axis (e.g., 1000 to convert s to ms)

    Returns:
        (x_vals, baseline_rms, nom_rms, ff_rms) as np arrays
    """
    rows = load_csv(csv_path)

    # Get baseline (single row with config=baseline)
    baseline_rows = [r for r in rows if r["config"] == "baseline"]
    if not baseline_rows:
        # If no explicit baseline row, use the nominal+perturbed value at
        # x=0 as baseline (it represents the unperturbed nominal case).
        baseline_rms = parse_float(
            next(r["rms_cm"] for r in rows
                 if r["config"] == "nominal_perturbed"
                 and parse_float(r[x_key]) == 0.0))
    else:
        baseline_rms = parse_float(baseline_rows[0]["rms_cm"])

    # Get all unique x values from perturbed runs
    x_vals_set = set()
    for r in rows:
        if r["config"] in ("nominal_perturbed", "ff_residual_perturbed"):
            x_vals_set.add(parse_float(r[x_key]))
    x_vals = sorted(x_vals_set)

    nom_rms = []
    ff_rms = []
    for x in x_vals:
        nom_row = next((r for r in rows
                        if r["config"] == "nominal_perturbed"
                        and parse_float(r[x_key]) == x), None)
        ff_row = next((r for r in rows
                       if r["config"] == "ff_residual_perturbed"
                       and parse_float(r[x_key]) == x), None)
        nom_rms.append(parse_float(nom_row["rms_cm"]) if nom_row else float("nan"))
        ff_rms.append(parse_float(ff_row["rms_cm"]) if ff_row else float("nan"))

    return (np.array(x_vals) * scale_x,
            baseline_rms,
            np.array(nom_rms),
            np.array(ff_rms))


def closure_pct(nom, ff, baseline):
    """Per-point gap closure: (nom - ff) / (nom - baseline) * 100."""
    pct = np.zeros_like(nom)
    for i, (n, f) in enumerate(zip(nom, ff)):
        gap = n - baseline
        recovered = n - f
        if abs(gap) > 0.05:
            pct[i] = recovered / gap * 100
        else:
            pct[i] = np.nan
    return pct


def plot_panel(ax, x, nom, ff, baseline, x_label, title,
               highlight_zone=None):
    """One panel of the aggregate figure."""
    ax.plot(x, nom, "o-", color="C3", lw=1.8, ms=5,
            label="Nominal MPC")
    ax.plot(x, ff, "o-", color="C0", lw=1.8, ms=5,
            label="Feedforward Residual MPC")
    ax.axhline(baseline, ls="--", color="C2", alpha=0.7,
                label=f"Unperturbed baseline ({baseline:.2f} cm)")
    if highlight_zone is not None:
        ax.axvspan(highlight_zone[0], highlight_zone[1],
                    color="C3", alpha=0.08,
                    label=highlight_zone[2] if len(highlight_zone) > 2 else None)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Tracking RMS error (cm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def main():
    print("=" * 60)
    print("Stage 5 — aggregate figure across all perturbations")
    print("=" * 60)

    # Load all four sweeps
    print("\nLoading sweeps...")
    x_m, base_m, nom_m, ff_m = aggregate_sweep(CSV_MASS, "mass_factor")
    print(f"  Mass:  {len(x_m)} levels, baseline = {base_m:.2f} cm")
    x_d, base_d, nom_d, ff_d = aggregate_sweep(CSV_DRAG, "drag_coeff")
    print(f"  Drag:  {len(x_d)} levels, baseline = {base_d:.2f} cm")
    x_l, base_l, nom_l, ff_l = aggregate_sweep(CSV_LAG, "lag_tau_s",
                                                scale_x=1000.0)
    print(f"  Lag:   {len(x_l)} levels, baseline = {base_l:.2f} cm")
    x_t, base_t, nom_t, ff_t = aggregate_sweep(CSV_DELAY, "delay_s",
                                                scale_x=1000.0)
    print(f"  Delay: {len(x_t)} levels, baseline = {base_t:.2f} cm")

    # Build the 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_panel(axes[0, 0], x_m, nom_m, ff_m, base_m,
               "Mass factor (× nominal)",
               "Mass perturbation — recovery 80–92% (architecture matched)")
    axes[0, 0].axvline(1.0, ls=":", color="gray", alpha=0.4)

    plot_panel(axes[0, 1], x_d, nom_d, ff_d, base_d,
               "Drag coefficient b",
               "Drag perturbation — recovery ~0% (architecture mismatched)")

    plot_panel(axes[1, 0], x_l, nom_l, ff_l, base_l,
               "Motor lag time constant (ms)",
               "Motor lag — neutral below cliff, destabilizing in transition zone",
               highlight_zone=(42.5, 49.5, "transition zone"))

    plot_panel(axes[1, 1], x_t, nom_t, ff_t, base_t,
               "Time delay (ms)",
               "Time delay — flat below cliff (15 ms), no smooth regime to test")

    fig.suptitle("Stage 5: feedforward residual recovery across perturbation types",
                  fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {OUT_PNG}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY — gap closure by perturbation type")
    print("=" * 70)

    for name, x, nom, ff, base, fmt, unit in [
        ("Mass",       x_m, nom_m, ff_m, base_m, "{:>5.2f}", "× nominal"),
        ("Drag",       x_d, nom_d, ff_d, base_d, "{:>5.2f}", ""),
        ("Motor lag",  x_l, nom_l, ff_l, base_l, "{:>5.0f}", "ms"),
        ("Time delay", x_t, nom_t, ff_t, base_t, "{:>5.0f}", "ms"),
    ]:
        print(f"\n{name} (unperturbed baseline = {base:.2f} cm)")
        pct = closure_pct(nom, ff, base)
        header = f"  {'value':>6}{unit:>10}  {'nominal':>10}  {'residual':>10}  {'closure':>10}"
        print(header)
        print(f"  {'-'*6}{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for i, xv in enumerate(x):
            c_str = "n/a" if np.isnan(pct[i]) else f"{pct[i]:>6.0f}%"
            val_str = fmt.format(xv)
            print(f"  {val_str:>6}{unit:>10}  "
                  f"{nom[i]:>10.2f}  {ff[i]:>10.2f}  {c_str:>10}")


if __name__ == "__main__":
    main()
