# Stage 2 — MuJoCo + Skydio X2 + Cascaded PD Baseline

## What this stage accomplished

A working cascaded PD controller for the Skydio X2 quadrotor in MuJoCo, validated on three increasingly demanding tasks: hover, position step, and circle trajectory tracking. This serves as the classical-control baseline against which the MPC implementation in Stage 3 will be compared.

## Files

- `src/x2_constants.py`         — physical constants extracted from the X2 model
- `src/cascaded_pd_controller.py` — controller (position + attitude PD + motor mixer)
- `scripts/01_view_x2.py`       — model load and viewer sanity check
- `scripts/02_inspect_x2.py`    — extracts mass, inertia, geometry, actuator ranges
- `scripts/03_hover_test.py`    — hover at (0, 0, 1) m
- `scripts/04_step_test.py`     — 2 m horizontal step
- `scripts/05_circle_test.py`   — 1.5 m radius horizontal circle at 1.57 m/s

## Verified results

| Test    | Metric                  | Value     |
|---------|-------------------------|-----------|
| Hover   | Final position error    | < 1 nm    |
| Hover   | Per-motor steady thrust | 3.250 N (matches mg / 4) |
| Step    | 5 cm settling time      | 1.60 s    |
| Circle  | RMS tracking error      | 0.022 m   |
| Circle  | Max tracking error      | 0.052 m   |

## Controller architecture

Cascaded structure (matches standard quadrotor practice):

  Position PD (outer)  -> desired total thrust + desired body z-axis
  Attitude PD (inner)  -> desired body torques (geometric, Lee et al. style)
  Motor mixer          -> four motor thrust commands

Position loop accepts optional v_des and a_des_ff for trajectory tracking;
defaults to zero (hover behavior) when not provided.

## Physical parameters (Skydio X2)

| Parameter       | Value                 |
|-----------------|-----------------------|
| Mass            | 1.325 kg              |
| Inertia (diag)  | 0.0607, 0.0365, 0.0254 kg m^2 |
| Arm x, y        | 0.14 m, 0.18 m        |
| Drag/thrust ratio (kappa) | 0.0201      |
| Max thrust/motor| 13 N                  |
| Hover thrust/motor | 3.25 N (25 % of max) |
| Default timestep | 0.01 s (100 Hz)       |

## What this baseline is not

- Not optimal. Pure PD with feed-forward; no anticipation, no constraints.
- Not robust to model mismatch. All gains assume the nominal mass and inertia.
- Limited to smooth, moderate-speed trajectories. Aggressive maneuvers
  would saturate motors (already observed briefly during the 2 m step).

These limitations motivate the MPC implementation in Stage 3.

## What is next

Stage 3 — nonlinear MPC formulation with CasADi, using the same Skydio X2 model, replacing the position loop only (the inner attitude loop and motor mixer carry over from this stage).
