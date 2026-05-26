# Stage 4 — Perturbations and Characterization of Model Mismatch

## What this stage accomplished

A characterization of how the Stage 3 nominal MPC degrades when the simulator's plant deviates from the MPC's prediction model. Five perturbation types are studied independently: mass, inertia, linear drag, motor lag, and pure transport delay. The MPC is unchanged across all experiments; only MuJoCo's plant is perturbed. The result is a quantitative map of where the nominal MPC remains usable, where it degrades gracefully, and where it fails catastrophically. This map is the baseline that Stage 5's learned residual will be measured against.

## Files

- `stage4_src/perturbations.py` — five perturbation utilities: functions for mass and inertia (modify model at load), classes for drag, motor lag, and time delay (applied dynamically during the simulation loop).
- `stage4_src/experiment_runner.py` — reusable trial runner. Takes a perturbation spec, runs the MPC on the standard circle trajectory for 15 seconds, returns RMS/max tracking error and stability flag.
- `scripts/01_test_perturbations.py` — sanity check that each of the five perturbation types is applied correctly and produces sensible behavior.
- `scripts/02_sweep_mass.py` — mass factor sweep (0.7 to 1.4).
- `scripts/03_sweep_inertia.py` — inertia factor sweep (0.5 to 3.0).
- `scripts/04_sweep_drag.py` — linear drag coefficient sweep (0 to 2.0 N s/m).
- `scripts/05_sweep_motor_lag.py` — motor lag time constant sweep (0 to 100 ms).
- `scripts/05b_sweep_motor_lag_zoom.py` — fine-grained sweep around the motor-lag cliff (30 to 65 ms).
- `scripts/06_sweep_time_delay.py` — time delay sweep (0 to 50 ms).
- `scripts/06b_sweep_time_delay_zoom.py` — fine-grained sweep around the time-delay cliff (10 to 17 ms).
- `scripts/07_aggregate_results.py` — combines all sweep CSVs and produces the aggregate figure.

## Verified results

Summary of tracking error (post-ramp) across all five perturbation types:

| Perturbation type | Nominal RMS | Worst stable RMS | Stable range | Crashed trials |
|---|---|---|---|---|
| Mass | 1.65 cm | 9.95 cm at factor 1.4 | factors 0.7 to 1.4 | 0 |
| Inertia | 1.65 cm | 4.13 cm at factor 0.5 | factors 0.5 to 2.0 | 1 (factor 3.0) |
| Drag | 1.65 cm | 13.10 cm at b = 2.0 N s/m | 0 to 2.0 | 0 |
| Motor lag | 1.65 cm | 12.30 cm at tau = 50 ms | 0 to 50 ms | 5 (55, 60, 65, 70, 100 ms) |
| Time delay | 1.65 cm | 1.65 cm at delay 14 ms | 0 to 14 ms | 9 (15 ms and beyond) |

The headline observation: the five perturbation types produce qualitatively different degradation patterns. Some degrade smoothly across a wide range (mass, drag); some have a binary nature (motor lag, time delay); some affect the controller only weakly (inertia, due to the kinematic-thrust formulation).

## Perturbation mechanisms

**Mass perturbation.** Scales MuJoCo's `body_mass[x2]` by a factor at model load. The MPC's `MASS = 1.325 kg` constant is unchanged. Mismatch effect: thrust commands are calibrated for the nominal mass, so the drone sags (heavier) or accelerates too aggressively (lighter) compared to MPC's prediction.

**Inertia perturbation.** Scales MuJoCo's `body_inertia[x2]` diagonal by a factor at model load. The MPC's prediction model has no inertia term (kinematic-thrust formulation). Mismatch effect: the body-rate inner loop's actual rotational dynamics differ from what MPC implicitly assumes (perfect rate tracking). Effect is weak because angular dynamics are absorbed by the inner loop's high bandwidth.

**Drag perturbation.** Applies an unmodeled linear drag force `F_drag = -b * v` at every physics step via `data.xfrc_applied`. The MPC's prediction model has no drag term. Mismatch effect: actual velocity is lower than predicted at high speed, creating tracking lag.

**Motor lag.** Each motor's actual thrust is a first-order filter of the commanded thrust, with time constant `tau`. The MPC assumes instantaneous thrust. Mismatch effect: high-frequency rate commands from MPC are smeared by the motor filter, the inner loop falls behind, and tracking degrades.

**Time delay.** Each motor command vector is queued and applied to MuJoCo with a fixed delay `tau_d`. The MPC assumes zero delay. Mismatch effect: classical closed-loop delay instability. The controller responds to state that's already obsolete.

## Experimental conditions

All perturbations are tested against the same nominal reference trajectory: a 1.5 m radius horizontal circle at 1.57 m/s tangential speed, with a 2 second smooth cubic ramp-in, total duration 15 seconds. Tracking error is computed over the post-ramp window (after t = 3 s). Each trial loads a fresh MuJoCo model to ensure perturbations don't accumulate across trials. Crash detection flags trials where the drone falls below 10 cm altitude, exceeds 50 m from origin, or exceeds 30 m/s speed.

## Aggregate figure

`aggregate_perturbations.png` shows all five sweeps in a single 2x3 grid. Each subplot has the perturbation magnitude on the x-axis and RMS tracking error on the y-axis. Crashed trials are shown as red X markers at the top of each subplot. The nominal RMS (1.65 cm) is shown as a horizontal reference line. This figure is the central visual deliverable of Stage 4.

## What is and is not in this stage

What is:
- A controlled study of how MPC tracking error scales with five distinct perturbation axes.
- A characterization of the boundaries where the controller transitions from graceful degradation to catastrophic failure.
- A reproducible harness that can be extended with new perturbation types.

What is not:
- A study of robust MPC formulations. The MPC is unchanged across all trials. We are measuring the *vanilla* controller's response to model mismatch, not designing for it.
- A study of compound perturbations. Each trial perturbs one axis at a time. Combined perturbations are out of scope for this stage.
- Real hardware. All experiments are in simulation against MuJoCo's plant.

## What is next

Stage 5 trains a small learned residual model on offline rollout data, integrates it into the MPC's prediction, and measures how much of the Stage 4 degradation it recovers. The five perturbation types map onto a spectrum of expected difficulty:

- Easy to learn: mass, drag (direct, predictable effects on velocity dynamics).
- Medium: motor lag transition zone (45-50 ms), where the controller is wounded but not dead.
- Hard: inertia at extremes, motor lag past the cliff.
- Out of reach: pure time delay (no graceful regime to learn from).

---

## Additional notes: how Stage 4 actually went

Notes on what happened during the data collection, kept for reference and possible discussion-section material.

### The harness came together quickly

After the conceptual setup (deciding to use a stateless trial runner that loads a fresh model each time, deciding to apply mass and inertia at load and drag/motor lag/time delay dynamically), the actual code was straightforward. The runner is about 150 lines and dispatches all five perturbation types from a single spec dict. The sweep scripts are 50-line templates that look near-identical to each other, which is the right kind of repetition — it's the data that varies, not the code.

The one near-bug worth noting: when adding motor lag and time delay, the simulation loop needed to thread thrust commands through the active filter *before* writing to `data.ctrl`, while drag still needed to be applied as an external force *after* the controller had computed its output. Mixing these up would silently produce wrong physics. The clean separation in the runner (thrusts get filtered, drag gets added as xfrc_applied) made this easy to get right.

### The five perturbation types behaved differently than I expected

Going in, I expected mass, inertia, and drag to be the "main" perturbations, with motor lag and time delay as supplementary ones. The data flipped this:

- Inertia turned out to be nearly inert across the whole range, because the MPC's kinematic-thrust formulation doesn't depend on inertia. The only meaningful effects appeared at the extremes (factor 0.5 and 3.0). Worth keeping as a perturbation type because the absence of effect is itself a finding: it teaches the reader something about the MPC formulation.
- Motor lag turned out to have a binary, on-cliff effect rather than a smooth degradation curve. The cliff happens to coincide with the controller's effective timescale (about 30-50 ms), which is set by the inner-loop bandwidth.
- Time delay turned out to be the most aggressively destabilizing perturbation by a wide margin. The transition from "perfect tracking" to "controller crashes" happens with a single millisecond of additional delay.

These three findings together actually strengthen the paper's narrative. Different perturbation types have different curves; some are continuously degradable and learnable, some are binary, some are formulation-dependent. The aggregate figure tells this story at a glance.

### The drag finding was the unexpected positive

When I ran the drag sweep, I noticed that b = 0.1 N s/m produced *better* tracking than the nominal (1.18 cm vs 1.65 cm). Initially I thought this was a bug, but on inspection it's a real effect: small drag provides passive damping that smooths out the residual oscillations in MPC's rate commands. The MPC's inner loop is the limiting factor on tracking; mild damping helps it. Past about b = 0.25, the drag becomes a disturbance the MPC can't predict, and the benefit reverses.

This is the kind of result that I would have rejected as "noise" without thinking carefully about it, but it has a clean physical interpretation. It belongs in the paper as a one-sentence observation: a small amount of unmodeled drag can *improve* tracking on smooth trajectories because it dampens controller-induced oscillation.

### The cliff zooms saved the figure

The original 5-script sweep gave us the gross structure of each curve but missed the transitions where the controller fails. For motor lag, the main sweep had a 20-ms gap between tau = 30 ms (nominal) and tau = 50 ms (severe). For time delay, the gap was 5 ms between delay = 10 ms (nominal) and delay = 15 ms (crash). Filling these in with two additional zoom sweeps (35 ms increments for motor lag, 1 ms increments for time delay) revealed the actual shape of the transitions:

- Motor lag: stable through 40 ms, transitional at 45 ms (5 cm RMS), severe at 50 ms (12 cm), crash at 55 ms and beyond. So the cliff has a finite transition zone of about 5-10 ms.
- Time delay: stable through 14 ms, immediate crash at 15 ms and beyond. The cliff is essentially zero-width.

Knowing the exact transition behavior matters for Stage 5: the controller's "useful" range for each perturbation is now precisely defined. The motor lag's 45-50 ms transition zone is where a learned residual has a chance to help; outside that zone there's nothing for the residual to fix or save.

### Things I would do differently

1. **Plan the cliff zooms from the start.** The original 5-script sweep was efficient but missed important detail. Including 2-3 finer points across each transition from the start would have saved having to write the zoom scripts as a follow-up.
2. **Use a single shared sweep template.** The five sweep scripts are 90% identical; a single function `run_sweep(perturbation_type, values, x_label, x_col_name)` would have been cleaner and the per-script code would be 5 lines instead of 80. The current per-script structure is fine for one-off use but doesn't generalize.
3. **Make the experiment runner record more state.** Currently it returns RMS, max, and a stable flag. Logging the full trajectory (position, velocity, body rates, MPC commands) for each trial would let us produce time-series plots showing the actual failure mode for any individual perturbation. We could go back and add this if Stage 5 needs it.

### What is real now

Stage 4 produced:
- A working perturbation harness for five distinct model-mismatch types.
- 47 simulation trials covering five perturbation axes, with the main behavior and cliff regions both characterized.
- A single aggregate figure (`aggregate_perturbations.png`) that summarizes the controller's response to model mismatch.
- CSV files with all the headline numbers for the paper's experimental table.
- A clear baseline against which Stage 5's learned residual will be measured.

The most important result for the project's narrative: the nominal MPC degrades smoothly under some perturbation types (mass, drag) and binary under others (motor lag, time delay). The next stage will study which of these are learnable with a small residual model.