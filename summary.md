# Project Refresher: `uav-mpc-learning`

*Written to bring you back to full speed on a project you built ~4–5 months ago in a ~2 month sprint, submitted to IEEE iCONNECT 2026 (BD), currently under review. This file is your memory prosthetic — read it top to bottom once, then use it as a reference.*

---

## 1. The one-paragraph version (for a non-engineer)

Imagine a delivery drone that has been taught to fly a perfect circle by a very precise internal "brain" that assumes it knows exactly how heavy the drone is, how the air pushes back on it, and how quickly its propellers respond to commands. In the real world, none of that is ever exactly right — the drone might be carrying a slightly heavier package than expected, wind might drag on it, or the motors might lag a few milliseconds behind commands. This project builds that internal brain (a modern control technique called Model Predictive Control, or MPC), then deliberately breaks its assumptions in a realistic physics simulator to see exactly how and when its flying gets worse. It then trains a small, fast "helper" neural network that watches the gap between what the brain expects and what actually happens, and tries to nudge the drone's thrust to compensate in real time. The interesting — and honest — finding is that this helper is not a magic fix: it recovers 80–92% of the lost accuracy when the problem is a *weight* mismatch, does essentially nothing when the problem is *air drag*, and — most surprisingly — actively makes things worse right before the drone would have crashed anyway because of *motor lag*. The paper's contribution isn't "we made it perfect," it's a careful, honest map of exactly where this kind of AI patch helps, does nothing, or backfires.

---

## 2. The project in engineering terms

### 2.1 What it is

A simulation-based study of **nonlinear Model Predictive Control (MPC) for quadrotor trajectory tracking under model mismatch**, and of whether a **small learned residual dynamics model**, integrated as a **feedforward thrust correction**, can recover the tracking performance lost when the real vehicle ("the plant") deviates from the controller's internal prediction model ("the nominal model").

Platform: **Skydio X2 quadrotor** (via the MuJoCo Menagerie, Apache-2.0 licensed reference model), simulated in **MuJoCo**. Optimization: **CasADi + IPOPT**. Learned component: a **5,641-parameter PyTorch MLP**, later hand-converted into a symbolic **CasADi function**.

### 2.2 Why this project / motivation

Nonlinear MPC is the modern standard for agile quadrotor trajectory tracking (cited lineage: Faessler et al. 2018 on differential flatness with rotor drag; Torrente et al. 2021 and Salzmann et al. 2023 on data-driven/neural MPC). But MPC is only as good as its internal model. Real drones drift from that model because of:
- **Payload / mass variation**
- **Aerodynamic drag** (significant above a few m/s)
- **Motor/ESC lag** (first-order actuator dynamics)
- **Sensing/command time delay**

The obvious "fix" people reach for is to bolt a learned residual model onto the controller. The literature has two ways to do this (in-constraint, inside the optimizer's dynamics; or feedforward, outside it), but — per the paper's own related-work framing — nobody had systematically characterized **which perturbation types a feedforward residual can actually fix**, as opposed to just picking one favorable perturbation and reporting a win.

### 2.3 Research statement (the actual question being asked)

> Given a learned residual model that accurately predicts one-step dynamics mismatch in simulation, how much of the closed-loop tracking degradation caused by mass, drag, motor lag, and time delay can a **feedforward residual architecture** recover — and what specifically limits that recovery?

Note the framing: the question is *not* "is the neural net accurate" (that's the easy, boring part — it's checked via held-out R²). The question is what happens when you actually close the loop with it.

### 2.4 Problem statement, restated technically

Given a nominal prediction model `f_nominal(x, u)` used inside an MPC's dynamics constraints, and a plant that actually evolves as `x_next = f_nominal(x, u) + r(x, u)` for some unknown, perturbation-dependent residual `r`, can a small learned approximation of `r`, applied as an **external, single-channel (vertical-thrust) correction after the MPC solve**, close the gap between nominal-MPC-on-perturbed-plant performance and nominal-MPC-on-unperturbed-plant performance (the "floor")?

### 2.5 The three contributions (verbatim intent from the paper)

1. **A reproducible perturbation characterization** of a baseline quadrotor MPC across 5 mismatch types (mass, inertia, drag, motor lag, time delay) in MuJoCo — with degradation curves and instability "cliffs" fully resolved (millisecond-resolution zoom sweeps around the cliffs).
2. **A diagnosed failure mode of in-constraint residual MPC**: embedding the learned residual directly inside the MPC's optimization constraints causes IPOPT to blow its iteration budget in closed-loop operation (even though the *same problem solved standalone/offline* converges fine) — a warm-start / Jacobian-conditioning interaction, not a bug in the network or the solver settings. This failure, and the ~8 different mitigation attempts that didn't fix it, is treated as a first-class result, not a discarded dead end.
3. **A perturbation-wise evaluation of the feedforward architecture that pivoted out of that failure**: 80–92% gap closure on mass (matched regime), ~0% on drag (mismatched regime — the model *knows* the answer but the single-channel controller can't *act* on it), and a **destabilization effect of 15–29% (up to 45% in the README's more granular numbers) *increase* in tracking error** in the narrow 43–49 ms transition zone of motor lag, immediately before the cliff to instability.

### 2.6 Headline numbers to have on tap

| Quantity | Value |
|---|---|
| Nominal MPC floor (unperturbed circle, RMS) | 1.65 cm |
| MPC vs cascaded PD, circle RMS | 1.65 cm vs 2.25 cm (26.8% better) |
| MPC vs PD, max error | 1.68 cm vs 3.66 cm (54.1% better) |
| Mean MPC solve time | 4.2 ms (real-time budget is 10 ms @ 100 Hz) |
| Residual MLP size | 5,641 parameters, 2×64 tanh hidden layers |
| Training data | 88,500 transitions, 59 rollouts, 21 perturbation configs, 7.9 min wall time |
| Training time | 28.6 s on CPU, 50 epochs, Adam, lr 1e-3 |
| Validation fit | R² > 0.97 on velocities, 0.81–0.88 on positions |
| Residual eval cost | ~26–27 µs per call (negligible vs the 4.2 ms MPC solve) |
| Mass sweep gap closure | 80–92% (stable range: mass factor 0.7–1.4) |
| Drag sweep gap closure | ≈ 0% (±2%), stable range 0–2.0 N·s/m |
| Motor lag: flat region | 0–42 ms, no effect either way |
| Motor lag: destabilization zone | 43–49 ms, residual *increases* error 15–45% |
| Motor lag: crash | ≥ 50 ms |
| Time delay: crash | ≥ 15 ms (essentially a zero-width cliff, nothing to learn) |

---

## 3. Codebase architecture

### 3.1 The full project tree

```
uav-mpc-learning/
├── README.md                          # top-level project pitch + stage table
├── requirements.txt                   # ⚠ see Section 4 — this is NOT a clean project-specific file
├── paper.tex                          # the IEEE iCONNECT paper source (⚠ lives at root, not in stage6-paper/)
├── collect_figures.py                 # utility: walks the repo, copies all image files into figures/
├── notes/
│   └── stage2_observations.md         # free-form working notes from Stage 2 debugging
├── videos/
│   ├── uav_mpc_demo.gif               # README hero demo
│   └── UAV Drone Sim with MPC.webm    # same, video form
├── figures/                            # nested, stage-prefixed dump from collect_figures.py
├── final_figures/                      # flat, paper-ready figure set (curated for LaTeX)
│
├── stage1-uav-fundamentals/           # THEORY — no code, just learning notes
│   ├── 00_README.md
│   ├── 01_paper_reference_summary.md  # Faessler et al. 2018 study notes
│   └── 02_chapter1_quadrotor_basics.md
│
├── stage2-mujoco-setup/               # BASELINE — classical PD controller + MuJoCo plumbing
│   ├── README.md
│   ├── scripts/
│   │   ├── 01_view_x2.py              # load model, sanity-check viewer
│   │   ├── 02_inspect_x2.py           # extract mass/inertia/geometry from the XML
│   │   ├── 03_hover_test.py
│   │   ├── 04_step_test.py
│   │   └── 05_circle_test.py
│   └── stage2_src/
│       ├── x2_constants.py            # ★ single source of truth for all physical constants
│       └── cascaded_pd_controller.py  # ★ PD baseline + the motor mixer (reused everywhere)
│
├── stage3-mpc-baseline/               # THE CORE CONTROLLER — nominal nonlinear MPC
│   ├── README.md
│   ├── scripts/
│   │   ├── 01_test_dynamics.py .. 06_compare_pd_vs_mpc.py
│   └── stage3_src/
│       ├── quadrotor_model_casadi.py  # ★★★ the 9-state symbolic dynamics + RK4 integrator
│       ├── mpc_controller.py          # ★★★ the CasADi/IPOPT MPC itself
│       ├── body_rate_controller.py    # ★ inner-loop P controller (rates → torques)
│       └── trajectory.py              # hover / step / circle reference generators
│
├── stage4-perturbations/              # THE STRESS TEST — inject model mismatch, measure damage
│   ├── README.md
│   ├── scripts/
│   │   ├── 02_sweep_mass.py, 03_sweep_inertia.py, 04_sweep_drag.py,
│   │   │   05_sweep_motor_lag.py (+05b zoom), 06_sweep_time_delay.py (+06b zoom),
│   │   │   07_aggregate_results.py
│   └── stage4_src/
│       ├── perturbations.py           # ★★★ defines all 5 perturbation mechanisms
│       └── experiment_runner.py       # ★ shared "run one trial, get RMS/crash" harness
│
├── stage5-learned-residual/           # THE LEARNING COMPONENT — residual model + integration
│   ├── README.md                      # ★ the most detailed, most important README in the repo
│   ├── data/                          # (gitignored normally; currently present & untracked) .npz rollouts
│   ├── models/                        # residual_best.pt, residual_final.pt (committed checkpoints)
│   ├── scripts/
│   │   ├── 01_collect_data.py, 02_train_model.py, 03_test_casadi_export.py,
│   │   │   04_test_residual_mpc.py (failed attempt), 04b_diagnose_residual.py,
│   │   │   04_feedforward_residual_test.py (the working smoke test),
│   │   │   05..08_sweep_*_residual.py, 09_aggregate_residual.py
│   └── stage5_src/
│       ├── data_collector.py          # ★★ rollout harness that logs (state,control,next,next_nominal)
│       ├── residual_dataset.py        # PyTorch Dataset wrapper over the .npz
│       ├── residual_model.py          # ★★★ the MLP architecture + normalization logic
│       ├── casadi_residual.py         # ★★★ PyTorch → CasADi weight-transplant
│       ├── mpc_with_residual.py       # ★★ the FAILED in-constraint architecture (kept as evidence)
│       ├── feedforward_residual_controller.py  # ★★★ the WORKING architecture
│       └── feedforward_experiment_runner.py    # evaluation harness for Stage 5 sweeps
│
├── stage6-paper/                      # ⚠ effectively empty (just .gitkeep) — see Section 4
└── .venv/                             # local Python virtual environment (not project content)
```

### 3.2 The six-stage narrative logic

The directory names are not arbitrary — they are the actual chronological build order, and each stage's code is a **dependency of the next**:

1. **Stage 1** — pure theory, no code. Read-once background so Stage 3's MPC formulation isn't mysterious.
2. **Stage 2** — get MuJoCo + the Skydio X2 working at all, and build a classical cascaded PD controller as the sanity-check baseline (hover / step / circle all validated to sub-millimeter/cm accuracy). Also where `x2_constants.py` — the physical parameters everyone downstream imports — was reverse-engineered from the MuJoCo XML.
3. **Stage 3** — replace the PD outer loop with a proper nonlinear MPC (CasADi + IPOPT), keeping the *same* inner body-rate loop and motor mixer from Stage 2 so the comparison isolates the outer-loop contribution. This is the **nominal controller** used everywhere after.
4. **Stage 4** — freeze the controller, start breaking the *plant*. Five independent perturbation mechanisms, each swept from "harmless" to "crashes," to build the empirical map of where MPC degrades gracefully vs. catastrophically.
5. **Stage 5** — the ML stage. Collect labeled (state, control) → residual data by re-running Stage 3's MPC on Stage 4's perturbed plants; train a tiny MLP to predict the mismatch; try to bolt it into the MPC two different ways; report both the failure and the success.
6. **Stage 6 / root** — writing it up. In practice the actual paper (`paper.tex`) and its figures ended up living at the repo root rather than inside `stage6-paper/`, which is why that folder is empty — see Section 4.

Every stage's `_src/` package is imported by name from later stages via manual `sys.path` manipulation (each file does `STAGE3_ROOT = ...; sys.path.insert(0, STAGE3_ROOT)` at the top) — there's no installed package / `setup.py`. This is a deliberate "linear pipeline of scripts," not a reusable library, which matches the project's nature as a research characterization rather than production flight software.

### 3.3 Deep dive: the files that are the actual heart of the project

#### `stage2-mujoco-setup/stage2_src/x2_constants.py` — the ground truth
Every physical number in the whole project (mass 1.325 kg, inertia diagonal, arm lengths 0.14 m / 0.18 m, per-motor max thrust 13 N, drag/torque coefficient κ = 0.0201) is read from here. It was populated by *inspecting the live MuJoCo model* (`02_inspect_x2.py`), not by looking up a datasheet — the comments explicitly say "verified empirically." This matters: the nominal MPC's internal model (in `quadrotor_model_casadi.py`) hard-codes `MASS = 1.325` and `GRAVITY = 9.81` separately rather than importing from here, which is a minor duplication but not a bug since both stages were built from the same source values.

#### `stage3-mpc-baseline/stage3_src/quadrotor_model_casadi.py` — the "physics brain" the MPC believes in
This is the symbolic model the entire optimization is built around. Key design choices, and *why*:

- **9-dimensional state** `[px, py, pz, vx, vy, vz, roll, pitch, yaw]`, **not** 12-D (position+velocity+quaternion+body rates) and **not** a reduced 6-D (position+attitude only). The reasoning: this is the standard **kinematic-thrust cascade** formulation (Faessler et al., Sun et al. 2022) — the *outer* MPC only needs to reason about the slow translational + orientation dynamics; body rates are treated as **direct control inputs** (`u = [T, ωx, ωy, ωz]`), not as state to be integrated through rotational dynamics (torque, inertia, angular momentum). That's deliberately delegated to the fast inner-loop `body_rate_controller.py`. Folding rotational dynamics into the MPC state would mean the optimizer has to reason about inertia and torque limits too — more nonlinear, slower to solve, and the paper's whole "inertia barely matters to tracking" finding (Section V) is actually evidence that this separation was the right call.
- **Euler angles, not quaternions**, for orientation. A quaternion would add a 4th state dimension plus a unit-norm equality constraint the optimizer has to satisfy at every shooting node — extra nonlinear constraints IPOPT has to fight with. Euler angles are 3 numbers, no extra constraint, and gimbal lock (the classic argument against Euler angles) is a non-issue here because the reference trajectory (a gentle circle) never approaches ±90° pitch/roll.
- **RK4 integration, not Euler**, for the *discrete* dynamics used inside the optimizer's shooting constraints (`rk4_step`). The paper is explicit about why: RK4's higher integration order keeps numerical integration error small, so that when the paper later measures "gap between predicted and actual state," that gap is attributable to genuine model mismatch (drag, lag, etc.) rather than to sloppy discretization. The MPC's own prediction step (Δt=0.05 s) is 5× coarser than the simulator's physics step (Δt=0.01 s) specifically so the simulator's own error stays well below the prediction's.
- **No drag, no motor lag, no delay, no imperfect body-rate tracking** in this model — and the docstring says so outright: *"These omissions are deliberate: they define the gap that the perturbations in Section V will widen."* This file is the intentionally-naive "textbook" model; Stage 4 exists to attack exactly what's missing here.

#### `stage3-mpc-baseline/stage3_src/mpc_controller.py` — the actual optimization
Built once (`_build_optimizer`) as a CasADi `Opti()` problem with `N=20` shooting nodes at `dt=0.05s` (1-second lookahead), then re-solved every control tick by just updating the two parameters (`x0_param`, `ref_param`) and warm-starting from the previous solution shifted by one step. Things worth remembering:

- **Why CasADi + IPOPT, not e.g. a custom QP solver or scipy.optimize**: the dynamics constraint is genuinely nonlinear (trig functions of roll/pitch/yaw multiplying thrust), so this is a nonlinear program (NLP), not a QP. CasADi gives exact symbolic Jacobians/Hessians of that nonlinear constraint for free via automatic differentiation, and its tight IPOPT binding is the de facto standard in the quadrotor-MPC literature the paper cites (Torrente et al., Salzmann et al. both use the same combo).
- **Cost weights** `Q = diag([100,100,100, 10,10,10, 1,1,1])`, `R = diag([0.01, 0.1,0.1,0.1])`, terminal weight `10×Q`. Position weighted 10× velocity, 100× attitude — reflecting that position error is the actual metric being optimized against, velocity error matters because it's the derivative of what you're tracking, and attitude error is lightly weighted because the inner loop is already enforcing it independently.
- **Control cost centered at hover**, `Δu = u - u_hover` where `u_hover = [mg, 0,0,0]` — not centered at zero. This biases the optimizer toward "do nothing extra," which is both better conditioned numerically and physically sensible (a quadrotor's natural resting control input is hover thrust, not zero thrust).
- **Deliberately loose solver tolerances** (`tol=1e-3`, `acceptable_tol=1e-1`, `acceptable_iter=5`) — a "good enough, fast" solve rather than a fully converged one, which is what makes the 4.2 ms mean solve time possible at 100 Hz. This looseness is exactly what later blows up when the residual is embedded in-constraint (see below).
- **Fallback on solver failure**: if `opti.solve()` raises, it returns hover thrust + zero rates rather than propagating garbage — a basic safety net so a bad solve doesn't fling the sim state to NaN.

#### `stage2-mujoco-setup/stage2_src/cascaded_pd_controller.py` — the motor mixer everyone shares
Even though this file's *PD position/attitude loop* is only used as the Stage 2/3 comparison baseline, its `motor_mixer()` method (X-configuration thrust/torque → 4 individual motor thrusts, inverting the mixing matrix) is reused **unchanged** by Stage 3, Stage 4, and Stage 5. This is why the paper can claim the MPC and residual comparisons "differ only in the outer loop" — the geometry-to-motor-command math is one function, called everywhere.

#### `stage4-perturbations/stage4_src/perturbations.py` — the "break the plant" toolbox
Five independent, composable perturbation mechanisms, each isolated to one physical channel:
- `apply_mass_perturbation` / `apply_inertia_perturbation`: one-time scalar multiply on `model.body_mass` / `model.body_inertia` at load time.
- `LinearDrag`: `F = -b·v` applied every physics step via MuJoCo's `xfrc_applied` (external force) interface — a direct force injection, bypassing the model's own force pipeline.
- `MotorLag`: a per-motor first-order filter, `T_actual += (dt/τ)(T_cmd - T_actual)` — models ESC/motor response lag.
- `TimeDelay`: a fixed-length `deque` buffer that delays the *whole 4-motor command vector* by a fixed number of steps, pre-filled with hover thrust so the first `delay_s` seconds aren't garbage.

Why these specific four (plus inertia): they map directly onto real, physically distinct failure modes (payload, aerodynamics, actuator dynamics, sensing/computation latency), and — importantly — each is injected at a *different point in the control-to-physics pipeline* (model parameter vs. per-step external force vs. actuator filter vs. command-buffer delay), which is what lets the paper later show that a single-channel (vertical-thrust-only) correction helps some and not others: the correction channel and the perturbation's "point of injection" either line up or don't.

#### `stage5-learned-residual/stage5_src/residual_model.py` — the learned component itself
A 2-hidden-layer, 64-neuron, **tanh**-activated MLP, `input (13 = 9 state + 4 control) → residual (9)`, with mean/std normalization baked into the `forward()` pass as registered buffers (so they save/load with the checkpoint automatically). Two choices the docstring is explicit about, both driven by the *downstream* consumer (IPOPT), not by ML best practice in the abstract:

- **Tanh, not ReLU**: when this network's output feeds into IPOPT's constraint Jacobian (the in-constraint attempt), ReLU's kink at zero is a non-smooth point, and interior-point solvers rely on smooth derivatives — ReLU literally caused `Maximum_Iterations_Exceeded` failures via thrashing near kinks. Tanh is smooth everywhere.
- **Clamped input std** (`min_input_std=0.1`): because all training data comes from *closed-loop* rollouts (not open-loop random exploration), several state dimensions have tiny variance — yaw std ≈ 0.008 rad, `vz` std ≈ 0.053 m/s — since the controller is actively suppressing them. Normalizing by raw std would amplify unit input perturbations on those axes by >100×, pushing the constraint-Jacobian condition number above 1e6 (numerically unsolvable for IPOPT). Clamping to 0.1 caps the amplification at 10× and keeps conditioning around 6×10³ — the appendix (A.8) works through this arithmetic explicitly.

Why only ~5,600 parameters, not a bigger network: this is explicitly a **CPU-only, real-time-budget project** ("Hardware: designed for processor-only execution" in the README) — the residual has to evaluate in microseconds inside a 10 ms control loop, and the physics being approximated (a smooth, near-linear perturbation-to-residual relationship for mass/drag) doesn't need more capacity. The paper reports it *works* — R² > 0.97 on velocities — which retroactively validates that bigger wasn't needed.

#### `stage5-learned-residual/stage5_src/casadi_residual.py` — bridging PyTorch and the optimizer
This is a manual weight-transplant: it walks the trained `nn.Sequential`, pulls out each `Linear` layer's `W`/`b` as numpy, and rebuilds the exact same forward pass using `casadi.mtimes` and `ca.tanh` on symbolic CasADi variables — because **CasADi's symbolic graph can't call into PyTorch directly**; IPOPT needs everything as CasADi-differentiable expressions. One extra detail worth remembering: the CasADi version additionally squashes the *normalized* output through an extra `tanh` and rescales by `3×residual_std` before adding back the mean — an output-bounding safety measure not present in the raw PyTorch model, added so the residual can never blow up to an unphysical value when the optimizer explores states far outside the training distribution. Verified to match the PyTorch model to 1.1e-7 (attributed to float32→float64 conversion), and evaluates in ~26 µs.

#### `stage5-learned-residual/stage5_src/mpc_with_residual.py` — the failed attempt (kept intentionally)
This is the "theoretically correct" architecture: the residual is added directly into the MPC's own dynamics constraint (`x[k+1] = f_nominal(x,u) + scale·residual_fn(x,u)`), so the optimizer can *anticipate* the mismatch over the whole horizon instead of just reacting to it one step at a time. It's fully implemented, with a `residual_scale` knob and visibly hardened solver settings (looser tolerances, `max_iter=300`, `mu_strategy="monotone"`, limited-memory Hessian, and a `try/except` that salvages `opti.debug.value()` as a "partial" solution rather than only falling back to hover). None of it fixed the closed-loop failure. This file (and `04_test_residual_mpc.py` / `04b_diagnose_residual.py` / `diagnostic_output.txt`) is deliberately **not deleted** — it's documentation of contribution #2 (Section 2.5 above): the standalone diagnostic converges fine (3→61 IPOPT iterations as residual scale goes 0→1), but the *closed-loop, warm-started* version doesn't, because each solve's warm-start comes from a possibly-not-fully-converged previous solve, and the residual's nonlinearity dominates the local quadratic approximation IPOPT builds at each iterate, causing trust-region steps to shrink without producing descent.

#### `stage5-learned-residual/stage5_src/feedforward_residual_controller.py` — the architecture that actually ships
The pivot, and the one behind every Section VIII number. It wraps an unmodified `MPCController` (Stage 3, untouched — the solver never even sees the residual):
1. Solve the nominal MPC normally → get `u_mpc = [T, wx, wy, wz]`.
2. Query the residual model once, *after* the solve, at `(x_current, u_mpc)`.
3. Take only `residual[5]` (the predicted one-step vertical-velocity mismatch, `r_vz`).
4. Convert it to a thrust delta: `ΔT = -mass · r_vz / sim_dt` (derived by inverting the near-hover vertical acceleration equation `v̇z ≈ T/m - g`).
5. Clip `ΔT` to ±5 N (≈1/3 of hover thrust — a safety bound that the paper notes never actually triggers within the training distribution, but *does* nearly saturate at −4 N in the motor-lag destabilization zone — that's the mechanism behind finding #3).
6. Add to the MPC's commanded thrust, clip to actuator bounds, hand off to the *same* body-rate loop and mixer as everywhere else.

Why **single-channel, vertical-thrust-only**, rather than correcting all 4 controls: this is called out in the paper explicitly as "a deliberate scope choice" — it isolates the simplest possible correction architecture so the *pattern* of where it helps/fails is attributable to the architecture's structure, not to some other confound. It's also why drag "does nothing": drag on this circular trajectory shows up almost entirely as `r_vx`/`r_vy` (horizontal), which the model predicts accurately (R²>0.97) but which this controller has no channel to act on — a clean, diagnosed negative result rather than a mysterious one.

### 3.4 "Why X, not Y" quick-reference

| Question | Answer |
|---|---|
| Why 9 states, not 12 (quaternion+rates) or 6 (position+attitude only)? | Kinematic-thrust cascade: body rates are a *control input*, not integrated state; Euler (3) beats quaternion (4) because it avoids a unit-norm constraint and gimbal lock never triggers on this trajectory. |
| Why CasADi/IPOPT, not a QP solver? | The dynamics are genuinely nonlinear (trig × thrust); this is an NLP, and CasADi gives free exact autodiff Jacobians into IPOPT. Standard in the cited literature. |
| Why RK4, not Euler integration? | Reduces integration error so the measured "prediction vs. plant" gap reflects real model mismatch, not discretization artifacts. |
| Why MuJoCo, not another simulator? | Deterministic generalized-coordinate physics, a documented per-step external-force hook (`xfrc_applied`) needed to inject drag, and a freely reusable, parameter-calibrated Skydio X2 reference model (MuJoCo Menagerie, Apache-2.0). |
| Why tanh, not ReLU, in the residual MLP? | ReLU's non-smooth kink caused IPOPT to thrash (`Maximum_Iterations_Exceeded`) when the residual sat inside the solver's constraint graph. |
| Why clamp the input normalization std at 0.1? | Closed-loop training data has near-zero variance on some axes (yaw, vz); unclamped normalization would blow up the constraint Jacobian's condition number past 1e6. |
| Why feedforward, not in-constraint, for the final evaluation? | In-constraint is theoretically better (anticipates the residual over the horizon) but empirically failed in closed loop under warm-started IPOPT — documented, not swept under the rug. Feedforward sidesteps the solver entirely. |
| Why correct only vertical thrust, not all 4 controls? | Deliberate minimal-scope design so the resulting success/failure pattern (matched vs. mismatched vs. destabilizing) is attributable to structure, not confounded by a more complex correction law. |
| Why a circular reference trajectory for everything? | Sustained curvature exercises all 3 position axes + the rate loop + actuator limits continuously, while still being feasible for the unperturbed baseline — a single, consistent stress test used identically across every stage. |
| Why such a small MLP (5,641 params)? | CPU-only, real-time budget (needs to run in microseconds inside a 10 ms control loop); the mass/drag-to-residual relationship is smooth enough that more capacity wasn't needed (validated post-hoc by R² > 0.97). |

---

## 4. Things worth double-checking before you rely on this repo further

These are observations from reading the actual files, not assumptions — flagged because you said "run the whole thing if necessary," and because a couple of them could matter for a paper currently under review.

1. **`paper.tex` appears to have duplicated sections.** Reading it end-to-end, "Section V — Perturbation Characterization" (with all its subsections: Mass, Inertia, Drag, Motor Lag, Time Delay, Summary) appears **twice in a row, verbatim**, and "Section VIII — Evaluation" (Protocol, Mass, Drag, Motor Lag, Aggregate View, Computational Cost) also appears **twice in a row, verbatim**, before the bibliography. This looks like a copy-paste/merge artifact in the `.tex` source rather than intentional. Since `paper.tex` is currently **untracked** in git (shown as `??` in `git status`) and sits at the repo root rather than inside the empty `stage6-paper/` folder, I can't tell whether this is the same file you actually submitted to iCONNECT, an in-progress re-draft, or a local recreation. **I have not touched this file.** If you want, I can deduplicate it (straightforward — delete the second copy of each block) — just confirm you want that change and that it's safe relative to whatever you actually submitted.
2. **`\includegraphics` paths in `paper.tex` are flat** (e.g. `figures/mujoco_drone.png`), but the actual `figures/` directory (populated by `collect_figures.py`) is **nested and stage-prefixed** (`figures/stage3-mpc-baseline/scripts/mujoco_drone.png`). The flat layout the paper expects matches `final_figures/` instead. There's no `\graphicspath` in the preamble to reconcile this. Compiling `paper.tex` as-is, from the repo root, will likely fail to find images unless you compile from inside (or symlink) `final_figures/`.
3. **`requirements.txt` looks like a dump of an unrelated, much larger environment** (full of ROS2/`ros2-*`, `rclpy`, `moveit`, `rqt-*` packages) rather than a clean list of this project's actual dependencies. It does contain `casadi==3.7.2`, `mujoco==3.6.0`, `numpy`, `scipy`, `matplotlib` — but **no `torch`/`pytorch` entry**, despite Stage 5 being built entirely on PyTorch. If you ever need to recreate the environment from scratch, this file alone won't do it.
4. **`stage6-paper/` is empty** (just a `.gitkeep`). The actual paper source, the `final_figures/` used by it, and `collect_figures.py` are all currently sitting uncommitted at the repo root — this looks like an in-progress "gather everything for submission" session that never got moved into its intended folder or committed. Worth deciding where this should permanently live.
5. `stage5-learned-residual/data/` currently contains the full rollout `.npz` files (including a 32.5 MB `combined.npz`) and shows as untracked — the Stage 5 README describes this directory as "(gitignored, regenerable)," so this data being present and untracked is consistent with that description, just noting it's there and not committed.

None of the above were changed — this is a read-only pass, per your instructions.

---

## 5. Status at last touch

Per `README.md`: **"Status: Stage 4 in progress"** — that line is stale (Stage 5 and the paper clearly exist and are further along), a normal side effect of the README not being updated as later stages landed. The real state, based on git log and file contents: Stages 1–5 are functionally complete (baseline validated, perturbations characterized, residual trained and evaluated both ways), and the project's current frontier is the **paper itself** — specifically the duplication issue in point 1 above and reconciling the figures paths in point 2, both of which are relevant if you're about to produce a camera-ready version or revise for review comments.
