# Stage 1 — UAV Fundamentals

## What this stage covers

This stage builds the conceptual foundation needed before any code is written. It covers the quadrotor as a physical system, the standard control architectures used to control it, the mathematical idea of
differential flatness, and the role of model mismatch — which is the central theme of this project.

The reference paper for this stage is:
Faessler, M., Franchi, A., and Scaramuzza, D., "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories," IEEE Robotics and Automation Letters, 2018.

## Why this stage exists

The UAV control literature has a steeper conceptual ramp than RL. The state vector, the cascaded control hierarchy, the attitude representations, and differential flatness all need to be understood before the MPC formulation in Stage 3 will make sense. Writing these chapters in advance prevents the same kind of debugging confusion that took weeks to resolve during the quadruped deployment.

The chapters here are the author's own learning notes. They are written to be re-readable months later when revisiting the project.

## Files in this stage

- `00_README.md` — this file
- `01_paper_reference_summary.md` — a structured background reference on control systems, quadrotor dynamics, and differential flatness, used as quick-lookup material while studying the Faessler paper. Not the author's own writing; included as a study aid.
- `02_chapter1_quadrotor_basics.md` — Chapter 1: quadrotor state and control vectors, coordinate frames, the equations of motion, the cascaded control hierarchy, and why quadrotor control is nonlinear and underactuated.
- `03_chapter2_differential_flatness.md` — Chapter 2: the concept of differential flatness, the flat outputs for a quadrotor, why flatness with rotor drag is the contribution of Faessler et al., and what this gives us as control engineers.
- `04_chapter3_model_mismatch.md` — Chapter 3: what model mismatch means in this context, the three concrete kinds of mismatch this project will study, and why a learned residual is a candidate solution.
- Additional chapters may be added as the stage progresses.

## How to use this stage

Read the chapters in order. Use the reference summary as a lookup resource when a specific term or method needs a refresher. Do not treat the chapters as final; they are working notes and will be revised as the project progresses.

## Exit criteria

Stage 1 is complete when:
1. Chapters 1, 2, and 3 are drafted.
2. The author can state, from memory, the quadrotor state vector, the cascaded control hierarchy, the flat outputs, and three concrete examples of model mismatch.
3. The toolchain check in Stage 2 begins.
