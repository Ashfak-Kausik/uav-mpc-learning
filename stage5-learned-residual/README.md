# Stage 5: Learned Residual Dynamics

## Goal

Train a neural network to predict the one-step dynamics mismatch between the nominal quadrotor model and the perturbed plant, then use the learned residual to recover tracking performance under the perturbation regimes characterized in Stage 4. The residual is a function `r(x, u)` that approximates `state_next_perturbed - state_next_nominal` over one simulator step (10 ms).

## Repository layout

The Stage 5 code is organized as follows.
stage5-learned-residual/
├── stage5_src/
│   ├── data_collector.py                  # rollout harness with logged residuals
│   ├── residual_dataset.py                # PyTorch Dataset over collected data
│   ├── residual_model.py                  # MLP architecture and normalization
│   ├── casadi_residual.py                 # PyTorch to CasADi conversion
│   ├── mpc_with_residual.py               # in-constraint residual MPC (failed)
│   ├── feedforward_residual_controller.py # feedforward correction (works)
│   └── feedforward_experiment_runner.py   # trial runner across perturbations
├── scripts/
│   ├── 01_collect_data.py                 # data collection across 21 perturbations
│   ├── 02_train_model.py                  # train the residual MLP
│   ├── 03_test_casadi_export.py           # verify PyTorch -> CasADi parity
│   ├── 04_test_residual_mpc.py            # in-constraint MPC test (record of failure)
│   ├── 04_feedforward_residual_test.py    # feedforward smoke test (mass=1.2)
│   ├── 04b_diagnose_residual.py           # diagnostic of IPOPT iteration behavior
│   ├── 05_sweep_mass_residual.py          # mass sweep evaluation
│   ├── 06_sweep_drag_residual.py          # drag sweep evaluation
│   ├── 07_sweep_lag_residual.py           # motor lag sweep evaluation
│   ├── 08_sweep_delay_residual.py         # time delay sweep evaluation
│   └── 09_aggregate_residual.py           # produces the headline four-panel figure
├── data/    (gitignored, regenerable)
└── models/  (committed; trained residual checkpoints)

## Pipeline summary

**Phase 1: data collection.** Ran the nominal MPC under 21 perturbation configurations (six mass factors, five drag coefficients, six motor lag values, four time delay values) with three random seeds each, logging state-control-residual transitions at the simulator timestep of 10 ms. Used an initial velocity perturbation of plus or minus 0.3 m/s in place of an initial phase shift in the reference trajectory, since the latter caused 59 of 63 rollouts to crash in early experiments. Four rollouts crashed at motor lag values near the cliff and were excluded. The remaining 59 rollouts produced 88,500 transitions in 7.9 minutes of wall time.

**Phase 2: residual model.** Trained a small MLP with two hidden layers of 64 neurons each, total 5,641 parameters, on the collected dataset. Used tanh activations rather than ReLU because the latter is non-smooth, which is a problem for any future use inside a gradient-based optimizer. Used input normalization by per-dimension training mean and standard deviation, with the standard deviation clamped to a minimum of 0.1 to prevent ill-conditioning on narrow-distribution dimensions such as yaw (training std 0.008) and vertical velocity (training std 0.053). Trained for 50 epochs with Adam at learning rate 1e-3, batch size 256, on CPU only in 28.6 seconds. Final validation MSE 3e-7 with per-dimension R squared above 0.97 on the three velocity components and 0.81 to 0.88 on positions.

**Phase 3: CasADi conversion.** Extracted the trained weights, biases, and normalization statistics, then rebuilt the forward pass symbolically using `casadi.mtimes` for linear layers and `casadi.tanh` for activations. The resulting CasADi function takes raw state and control and returns the raw residual, with normalization folded into the function. Verified parity with the PyTorch model to within 1.1e-7 max absolute difference, attributable to float32 to float64 conversion. Per-evaluation time was 26 microseconds.

**Phase 4a: in-constraint residual MPC (attempted, failed).** Embedded the CasADi residual function in the MPC dynamics constraint as `x[k+1] = f_nominal(x[k], u[k]) + residual_fn(x[k], u[k])` and solved with IPOPT. The optimizer hit `Maximum_Iterations_Exceeded` on every solve in the production loop despite multiple rounds of mitigation: switching activations from ReLU to tanh, clamping the input normalization standard deviation, loosening solver tolerances to `tol = 1e-2` and `acceptable_tol = 10.0`, raising `max_iter` to 300, switching to a monotone mu strategy with small mu_init, using limited-memory Hessian approximation, and adding a residual scale parameter. A standalone diagnostic confirmed the solver could converge in isolation in 61 iterations and 4.3 seconds at full residual scale, but in the closed loop the warm-started solves repeatedly exceeded the iteration cap. The diagnostic transcript is saved at `scripts/diagnostic_output.txt` and the failed test at `scripts/04_test_residual_mpc.py`. Both are kept in the repository as documentation of the failure mode rather than removed.

**Phase 4b: feedforward residual controller (works).** Pivoted to a feedforward architecture. The nominal MPC of Stage 3 solves with its original dynamics and no residual in its constraints. After the solve, the residual model is queried once at the current state and the MPC's first control, then the predicted vertical velocity residual is converted into a thrust correction via `delta_thrust = -mass * residual_vz / sim_dt` and applied to the MPC's thrust before sending the four-motor commands to MuJoCo. The MPC sees no residual and faces no solver difficulty. The correction is clipped to plus or minus 5 N as a safety bound. This architecture is implemented in `stage5_src/feedforward_residual_controller.py`.

## Evaluation sweeps

Each perturbation type was swept across the same range used in Stage 4, with two trials per level: the nominal MPC under the perturbation, and the feedforward residual controller under the perturbation. The unperturbed nominal MPC tracks the same circle at 1.65 cm RMS and serves as the reference floor. Per-perturbation results below.

### Mass perturbation (recovery 80 to 92 percent)

The mass perturbation is the architecture's matched case. Mass mismatch shows up as a vertical-velocity offset, which is exactly the channel the feedforward correction acts on. The thrust correction tracks the perturbation almost linearly: about plus 1 N of thrust per 0.1 increase in mass factor. The correction is essentially zero at the nominal mass factor of 1.0, confirming that the residual does not introduce a bias when the dynamics match the nominal model.

| Mass factor | Nominal RMS (cm) | Residual RMS (cm) | Gap closure |
|---|---|---|---|
| 0.7 | 7.77 | 2.64 | 84 percent |
| 0.8 | 5.37 | 2.10 | 88 percent |
| 0.9 | 3.09 | 1.77 | 92 percent |
| 1.0 | 1.65 | 1.64 | n/a |
| 1.1 | 2.87 | 1.75 | 92 percent |
| 1.2 | 5.11 | 2.07 | 88 percent |
| 1.3 | 7.51 | 2.60 | 84 percent |
| 1.4 | 9.95 | 3.34 | 80 percent |

### Drag perturbation (no recovery)

Drag is the architecture's mismatched case. Linear drag opposes velocity in all axes, but on the circle trajectory the drone's vertical velocity is near zero by design, so the residual on the corrected channel is small. The mean thrust correction across all drag levels is under 0.05 N in magnitude. The residual model itself does predict horizontal velocity residuals with R squared above 0.97, but the controller in this study does not act on them.

| Drag coefficient | Nominal RMS (cm) | Residual RMS (cm) | Gap closure |
|---|---|---|---|
| 0.0 | 1.65 | 1.64 | n/a |
| 0.1 | 1.18 | 1.17 | minus 1 percent |
| 0.3 | 1.48 | 1.48 | minus 2 percent |
| 0.5 | 2.74 | 2.73 | 0 percent |
| 1.0 | 6.27 | 6.26 | 0 percent |
| 1.5 | 9.75 | 9.75 | 0 percent |
| 2.0 | 13.10 | 13.11 | 0 percent |

### Motor lag (neutral below the cliff, destabilizing in the transition zone)

The Stage 4 characterization showed motor lag has a flat tracking region from 0 to about 42 ms, then a narrow transition zone from 43 to 49 ms with rapid degradation, then crashes at 50 ms and above. In the flat region the residual is neutral, as expected, since there is no tracking error to recover. In the transition zone the residual makes the tracking error worse rather than better. At 44 ms the residual nearly doubles the nominal error from 3.17 cm to 4.60 cm; at 45 to 49 ms the residual adds 18 to 43 percent above the already-degraded nominal error. The mean thrust correction in this zone reaches minus 4 N, near the safety clip.

The mechanism is interpretable. Under severe motor lag the actual thrust trails the commanded thrust, so the drone tends to fall more than the nominal model predicts. The residual sees the vertical velocity discrepancy and responds by *reducing* commanded thrust to try to anticipate the predicted next-step state, but in this regime the right action is the opposite, namely to command even more thrust to compensate for the lagging motors. The residual extrapolates a correction that was valid for the smooth training regime into a transition regime where the underlying closed-loop assumption no longer holds. This is a clear failure mode of feedforward residual learning: it is reliable inside the smooth training regime and unreliable on the edges of stability.

| Lag (ms) | Nominal RMS (cm) | Residual RMS (cm) | Gap closure |
|---|---|---|---|
| 0 to 42 | 1.65 | 1.64 to 1.98 | n/a (flat) |
| 43 | 1.65 | 1.98 | n/a |
| 44 | 3.17 | 4.60 | minus 93 percent |
| 45 | 5.14 | 6.65 | minus 43 percent |
| 46 | 6.79 | 8.39 | minus 31 percent |
| 47 | 8.26 | 10.02 | minus 26 percent |
| 48 | 9.73 | 12.09 | minus 29 percent |
| 49 | 11.08 | 12.76 | minus 18 percent |

### Time delay (no smooth regime to evaluate)

Time delay produced an immediate crash at 15 ms in Stage 4, with completely flat tracking below 15 ms. There is no smooth degradation regime for the residual to address in this perturbation type. All measured residual results below the cliff match the unperturbed baseline.

| Delay (ms) | Nominal RMS (cm) | Residual RMS (cm) |
|---|---|---|
| 0 | 1.65 | 1.64 |
| 5 | 1.65 | 1.64 |
| 10 | 1.65 | 1.64 |
| 14 | 1.65 | 1.64 |

## Summary of findings

The feedforward residual recovers most of the tracking error in the perturbation regime that matches its correction channel (mass) and either is neutral or actively destabilizes in the regimes that do not. Three findings define the characterization:

1. **Matched, smooth degradation** (mass): the residual recovers 80 to 92 percent of the gap across the entire tested range. The thrust correction is interpretable and tracks the perturbation linearly, with no bias at the nominal operating point.

2. **Mismatched, smooth degradation** (drag): the residual is correctly predicting horizontal-velocity residuals but the single-axis controller cannot use them. Recovery is approximately zero across all drag values.

3. **Cliff-edge perturbations** (motor lag, time delay): in the flat region before the cliff there is no error to recover and the residual is neutral. For motor lag, the narrow transition zone before the cliff sees the residual extrapolate an incorrect correction, *increasing* the tracking error by 18 to 93 percent.

The architectural lesson is that feedforward residual correction is most effective when the perturbation manifests on a channel the controller can act on and operates in a regime that resembles the training distribution. Multi-axis correction using the predicted horizontal residuals is a natural extension and would likely improve recovery for drag and possibly for motor lag, at the cost of greater interaction with the body-rate inner loop. We leave this extension to future work.

## Headline figure

The four sweeps are summarized in `scripts/stage5_aggregate.png`, produced by `09_aggregate_residual.py`. The figure shows nominal MPC and feedforward residual MPC tracking RMS as a function of perturbation strength, alongside the unperturbed baseline reference, for each of the four perturbation types.
