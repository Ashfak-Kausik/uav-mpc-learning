# uav-mpc-learning

UAV trajectory tracking with Model Predictive Control (MPC) and a learned residual model under model mismatch. The project implements a clean nonlinear MPC for a quadrotor (Skydio X2 in MuJoCo, via CasADi) and characterizes how a small learned residual closes the gap between the nominal MPC model and a perturbed plant.

## Motivation

Nominal MPC assumes a perfect dynamics model. Real quadrotor systems exhibit model mismatch from payload variation, aerodynamic effects, and manufacturing tolerances. This project measures, in a controlled simulator setting, how tracking performance degrades under model mismatch and how much of that gap a small CPU-trainable residual model can recover.

## Approach

1. Implement a nominal MPC for quadrotor trajectory tracking in MuJoCo.
2. Define a parametric grid of model perturbations (mass, inertia, drag).
3. Measure tracking error envelopes for the nominal controller.
4. Train a small MLP residual model on offline rollout data.
5. Compare nominal MPC, gain-augmented MPC, and MPC with learned residual under matched conditions.

## Project Structure

| Stage | Directory | Purpose |
|-------|-----------|---------|
| 1 | `stage1-uav-fundamentals/` | Theory notes on quadrotor dynamics, control hierarchy, MPC fundamentals |
| 2 | `stage2-mujoco-setup/` | MuJoCo + Skydio X2 setup and a classical baseline controller |
| 3 | `stage3-mpc-baseline/` | Nominal nonlinear MPC implementation (CasADi) |
| 4 | `stage4-perturbations/` | Parametric perturbation harness and characterization |
| 5 | `stage5-learned-residual/` | Residual model training and integration |
| 6 | `stage6-paper/` | Paper drafts, figures, and submission materials |

Auxiliary notes and references live in `notes/`.

## Hardware

This project is designed for processor-only execution. No discrete GPU is required.

## Status

Stage 4 in progress.

## License

To be added.
