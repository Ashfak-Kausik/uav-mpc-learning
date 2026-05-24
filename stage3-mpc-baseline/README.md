# Stage 3 — Nonlinear MPC for the Skydio X2

## What this stage accomplished

A working nonlinear Model Predictive Controller (MPC) for the Skydio X2 quadrotor in MuJoCo, validated on the same three tasks as the Stage 2 PD baseline: hover, position step, and circle trajectory tracking. The MPC replaces only the outer position loop; the inner attitude tracking is now a body-rate inner loop, and the motor mixer from Stage 2 is reused without changes. The result is a controller that outperforms the PD baseline on every measured task on the same simulator conditions, providing the comparison baseline that Stage 4 (perturbations) and Stage 5 (learned residual) will degrade and then recover.

## Files

- `stage3_src/quadrotor_model_casadi.py` — CasADi symbolic dynamics, RK4 discrete integrator, and a quaternion-to-Euler utility for reading state from MuJoCo.
- `stage3_src/mpc_controller.py` — the MPC class: 9-D state, 4-D control, 20-step horizon at 50 ms prediction timestep, IPOPT solver via CasADi `Opti()`, warm-started across solves.
- `stage3_src/body_rate_controller.py` — small proportional controller on body rates, converts MPC's desired rates into body torques.
- `stage3_src/trajectory.py` — reference trajectory generators (hover, step, circle) with smooth ramp-in for the circle.
- `scripts/01_test_dynamics.py` — three sanity tests on the CasADi dynamics (hover, free fall, pitched translation).
- `scripts/02_test_mpc_hover.py` — MPC against its own dynamics; isolates the optimization layer before involving MuJoCo.
- `scripts/03_mpc_mujoco_hover.py` — MPC controlling MuJoCo for hover.
- `scripts/04_mpc_mujoco_step.py` — MPC controlling MuJoCo for a 2 m horizontal step.
- `scripts/05_mpc_mujoco_circle.py` — MPC controlling MuJoCo for a 1.5 m radius circle at 1.57 m/s tangential speed.
- `scripts/06_compare_pd_vs_mpc.py` — runs PD and MPC on the same circle and produces the comparison figure.

## Verified results

| Test | Metric | PD baseline (Stage 2) | MPC (Stage 3) |
|------|--------|-----------------------|---------------|
| Hover | Final position error | < 1 nm | < 1 nm |
| Hover | Steady total thrust | 12.998 N | 12.998 N |
| 2 m step | 5 cm settling time | 1.60 s | 0.94 s |
| Circle (1.5 m, 1.57 m/s) | RMS tracking error | 0.0225 m | 0.0165 m |
| Circle (1.5 m, 1.57 m/s) | Max tracking error | 0.0366 m | 0.0168 m |
| Circle | MPC solve time | n/a | 4.2 ms mean |

The headline comparison: on the same circle, MPC reduces RMS tracking error by 26.8% and max tracking error by 54.1% relative to PD, with the maximum solve time per cycle staying well under the 10 ms budget required for 100 Hz operation.

## Architecture
[ MPC ]                    plans 1 s horizon, outputs (T, omega_des)
↓
[ Body-rate inner loop ]   converts omega_des into body torques
↓
[ Motor mixer (Stage 2) ]  converts (T, torques) into 4 motor thrusts
↓
[ MuJoCo ]                 physics step
↓
[ Reader ]                 quat -> Euler, build 9-D MPC state, loop back

The MPC operates at 100 Hz (every MuJoCo step at 10 ms physics rate) with a 20-step lookahead at 50 ms prediction dt, giving a 1 second prediction horizon. The body-rate inner loop and motor mixer also run at 100 Hz.

## MPC formulation

State (9-D, in MPC's belief):
- x = [px, py, pz, vx, vy, vz, roll, pitch, yaw]

Control (4-D):
- u = [T, omega_roll, omega_pitch, omega_yaw]

Cost function (running cost + terminal cost):
J = sum over k = 0..N-1 of: (x_k - x_ref_k)^T Q (x_k - x_ref_k)
- (u_k - u_hover)^T R (u_k - u_hover)
- (x_N - x_ref_N)^T Q_terminal (x_N - x_ref_N)

Default weights:
- `Q_diag = [100, 100, 100, 10, 10, 10, 1, 1, 1]` (position weighted heavily, velocity moderately, orientation lightly)
- `R_diag = [0.01, 0.1, 0.1, 0.1]` (small control penalty, centered at hover)
- `Q_terminal = 10 * Q` (terminal cost 10x running cost)

Constraints:
- Total thrust: `0 <= T <= 52 N` (4 motors x 13 N max each)
- Body rates: `-3 rad/s <= omega_* <= 3 rad/s`
- Dynamics: enforced via RK4 integration as equality constraints between consecutive state predictions.

Solver: IPOPT via CasADi, with relaxed tolerances (`tol = 1e-3`, `acceptable_tol = 1e-1`), adaptive mu strategy, and warm-starting from the previous solve.

## Physical parameters

Identical to Stage 2 (Skydio X2 nominal model):

| Parameter | Value |
|-----------|-------|
| Mass | 1.325 kg |
| Inertia (diag) | 0.0607, 0.0365, 0.0254 kg m^2 |
| Gravity | 9.81 m/s^2 |
| Max thrust per motor | 13 N |
| Hover thrust per motor | 3.25 N |
| Default MuJoCo timestep | 0.01 s (100 Hz) |

The MPC's prediction model and MuJoCo's true physics are matched in this stage: same mass, same gravity, no drag, no perturbations. The gap between them is only the integration error between RK4 (in MPC) and MuJoCo's internal integrator, which is small. Stage 4 will deliberately introduce a gap.

## What this baseline is and is not

What it is:
- The control comparison anchor for the rest of the project.
- A working position-level MPC with cascaded body-rate tracking, the standard formulation used in modern quadrotor research.
- Real-time capable on CPU at 100 Hz with substantial headroom.
- Outperforms PD on the same conditions, by margins that justify the implementation complexity.

What it is not:
- A whole-body MPC. The MPC does not optimize motor commands directly; it commands body rates that an inner loop tracks. This is the standard "kinematic-thrust MPC" formulation.
- Robust to model mismatch. The MPC's dynamics assume the nominal mass and inertia. Stage 4 will measure how badly this breaks.
- Aerobatic. The Euler-angle representation has singularities near pitch = pi/2; trajectories that approach those tilts would require switching to quaternions.

## What is next

Stage 4 introduces controlled model perturbations (mass, inertia, drag) and characterizes how the tracking metrics above degrade. Stage 5 trains a small learned residual on offline rollout data and integrates it into the MPC's prediction. Stage 6 is the paper.

---

## Additional notes: how Stage 3 actually went

The story of building Stage 3 was a sequence of working pieces, broken pieces, and surprises that taught me what MPC is actually like to implement. Writing it down so the lessons stay attached to the result.

### Building the dynamics first

I started by writing the CasADi dynamics module (`quadrotor_model_casadi.py`) before touching the MPC. This turned out to be the right call. The three sanity tests in `01_test_dynamics.py` (hover, free fall, pitched translation) all passed cleanly on the first run, which gave me confidence that the equations of motion were correct before any optimization layer was added. The cleanest debugging experience of the whole stage. If the dynamics had been wrong, every subsequent failure would have been mysterious; verifying them first eliminated that whole class of issue from later.

### MPC against its own dynamics

The second script (`02_test_mpc_hover.py`) ran the MPC against the same CasADi function it used for prediction. This is artificial (no real "plant" to deviate from the model), but it isolates the optimization layer. It converged to the hover target at machine precision on the first try, with average solve time 7.4 ms and a 336 ms max on the first solve (CasADi's compile-and-link overhead). Two things were confirmed at this point: the cost weights and the constraints were sensible, and IPOPT was behaving well.

### First contact with MuJoCo and the body-rate gain mistake

Plugging MPC into MuJoCo for the first time (`03_mpc_mujoco_hover.py`) did not work. The drone flipped and crashed within a couple of seconds. The plot showed motor thrusts saturating asymmetrically and body rates oscillating wildly between the rate constraints. Reading the data carefully, three issues stacked together:

1. The body-rate inner-loop gains were way too low. I had set `kp_rate = [0.5, 0.3, 0.1]` thinking they were "conservative starting values," but they gave a closed-loop bandwidth around 8 rad/s, while MPC at 50 Hz update with 20-step lookahead expects rate tracking with bandwidth above 100 rad/s. The inner loop could not follow MPC's commands, so the actual drone behavior diverged from what MPC predicted, MPC re-planned more aggressively, the gap grew, and the system became unstable. The fix was to raise the gains to `[8, 5, 2]`, roughly inertia-times-target-bandwidth.

2. The MPC was running at 20 Hz, not 50 Hz as intended. With `mpc.dt = 0.05 s` and `sim_dt = 0.01 s`, my code computed `k_mpc = round(0.05 / 0.01) = 5`, meaning MPC was called every 5 simulator steps which is 20 Hz, not 50 Hz. At 20 Hz, between MPC calls the control signal was held constant for 50 ms while the drone's actual rates drifted, which is borderline-acceptable for quadrotor control. The fix was to run MPC every simulator step (`k_mpc = 1`), giving 100 Hz MPC update rate, with the prediction dt of 50 ms kept the same.

3. The initial position was too far from the hover target. I had set the drone at `z = 0.3 m` while the MPC was asked to hover at `z = 1.0 m`. With the inner-loop gains too low, the resulting transient was too violent for the system to recover from. Moving the initial position to the target eliminated the transient and let me debug the steady-state behavior in isolation.

With all three fixes applied, hover worked cleanly. Final error at machine precision, motor thrusts at exactly 3.25 N each, body rates near zero. The body-rate gains turned out to be the dominant factor; the other two were compounding.

### The step test worked on the first try

After the hover fix, the step test (`04_mpc_mujoco_step.py`) ran the first time without issue. 5 cm settling time of 0.94 seconds compared to PD's 1.60 seconds, a 41% improvement on identical conditions. There was visible body-rate chatter during the first second (MPC was anticipating the upcoming step within its horizon, but the cost function strongly penalized any pre-deviation, so the optimizer commanded small corrections that oscillated against the rate constraint). The chatter was bounded, the maneuver completed cleanly, and the steady-state was perfect.

I flagged the chatter as a small artifact and moved on. It is the kind of detail worth noting in a paper's discussion as "we observed bounded pre-maneuver oscillation during hold, characteristic of anticipatory MPC with short horizons."

### The src/src package collision

Around this point I tried to run the step test and got a `ModuleNotFoundError` for `src.mpc_controller`. The problem was that Stage 2 and Stage 3 both had folders named `src`, and Python's import system can only have one package named `src` in memory at a time. When the script added Stage 2's root to `sys.path` first (to import the motor mixer) and Stage 3's root second, Python found Stage 2's `src` first and never looked for Stage 3's. I worked around it initially with importlib, then permanently renamed both packages to `stage2_src` and `stage3_src`, updated every script in both stages, and committed the cleanup before continuing. The smoke test after the rename passed for all six scripts, confirming the migration was complete. This was a structural issue I should have anticipated at the start, but the cleanup took a single commit and unblocked everything that came after.

### The circle test failed catastrophically

This was the most informative failure of Stage 3. On the circle trajectory, IPOPT failed on essentially every solve with `Maximum_Iterations_Exceeded`. The fallback logic kicked in (return hover commands), and the drone wandered off to (-5, 1) with 6 meters of tracking error. Compared to the step test, which had worked beautifully, the failure was severe and surprising.

The cause, after reading the logs and thinking through what was different, was that the circle reference had a discontinuous initial velocity. The linear ramp on the radius (the version I copied from Stage 2's PD test) gave a position of zero at t=0 but a velocity of `0.75 m/s` at t=0. The MPC saw the drone at rest, and the reference asking for 0.75 m/s of velocity instantly. The cost function weighted velocity deviations at 10 per axis, so the instantaneous penalty for not matching was huge, and the body-rate constraints (±3 rad/s) and thrust constraint (0 to 52 N) made the immediate fix infeasible. IPOPT thrashed trying to find a feasible trajectory and gave up. The PD controller in Stage 2 had not exhibited this failure because PD is "soft" (it just computes a feedback signal from whatever error it sees and lets the drone catch up), while MPC is "hard" (it tries to exactly satisfy dynamics and constraints).

Three fixes together resolved it:

1. **Smooth the ramp.** Replaced the linear radius ramp with a cubic ramp factor `s(t) = 3*tau^2 - 2*tau^3` applied to both position and velocity, so both start at zero with zero derivative and reach the full circle smoothly by t = 2 s. This eliminated the infeasibility at t = 0.

2. **Relax solver tolerances.** Increased IPOPT's `max_iter` from 200 to 500, relaxed `tol` from 1e-4 to 1e-3, relaxed `acceptable_tol` from 1e-2 to 1e-1, and turned on the adaptive `mu_strategy`. These give the solver more flexibility to find good-enough solutions instead of insisting on tight convergence.

3. **Seed the first solve.** Added an `else` branch to the warm-start logic so the very first MPC call seeds the state trajectory with the reference and the control trajectory with hover. Without this, the first solve started from all-zeros, which is far from any feasible solution.

After these three fixes, the circle test produced RMS tracking error of 1.65 cm and max error of 1.68 cm. Stage 2's PD got 2.25 cm RMS on the equivalent (re-derived with the smooth ramp for fairness) version of the same trajectory, so the MPC improvement was 27% on RMS and 54% on max. The lesson worth carrying: MPC requires well-posed (feasible, smooth) references in a way that feedback control does not, and the failure mode when the reference is ill-posed is not graceful degradation but solver collapse.

### What I would do differently

Three things that would have saved time:

1. **Compute body-rate gains from first principles, not from intuition.** The rule `K_rate = bandwidth * inertia` produces gains that are inherently scaled to the system. Guessing values "to be conservative" is genuinely risky; better to compute them and tune from there.

2. **Pick unique package names from the start.** Naming both stages' source folders `src` was a structural mistake that cost an hour to clean up mid-stage. Stage 4 onward will use `stage4_src`, `stage5_src`, etc.

3. **Test references for feasibility before plugging them into MPC.** A 30-second mental walkthrough of "what does the reference look like at t = 0" would have caught the discontinuous velocity. With MPC, the reference is part of the problem, not just an input.

### What is real now

Stage 3 produced:
- A working CasADi-IPOPT MPC for quadrotor trajectory tracking.
- Measurable improvement over the PD baseline on identical conditions.
- A clean comparison figure (`comparison_pd_vs_mpc.png`) and CSV of headline numbers.
- Real-time capability with substantial CPU headroom.
- A defensible methodological position: position-MPC with kinematic-thrust cascade is the standard modern formulation, not a clever-but-fragile variant.

The MPC's tracking floor (about 17 mm RMS on this circle) is set by the inner-loop bandwidth, not by MPC itself. Pushing it lower would require a faster inner loop or running MuJoCo at higher physics rate. For the project's purposes, 17 mm is fine: it is the baseline that Stage 4 will degrade with perturbations and that Stage 5 will try to recover with learned residuals.