# Stage 2 — Working Observations

Personal notes from getting the cascaded PD baseline running. Captured
while it's fresh, for reference during paper drafting.

## On the Skydio X2 model itself

- The model has 1.325 kg mass, max thrust 13 N per motor, and a hover
  point at 3.25 N per motor (25 % of max). Healthy headroom; the drone
  has roughly a 4:1 thrust-to-weight ratio, which is on the agile end
  for a small quadrotor.
- Inertia is *not* symmetric across all three axes: Ixx (roll) is larger
  than Iyy (pitch), and Izz (yaw) is the smallest. This is consistent
  with an X-frame quadrotor where mass distribution is more spread along
  the body x-axis than along y. Worth remembering for the attitude
  controller tuning: the same gain on roll and pitch will produce
  different effective bandwidths.
- The model uses motor-level commands directly in Newtons. Each
  actuator's `gear[5]` encodes the drag-to-thrust ratio (kappa = 0.0201)
  and alternates sign between motors 1, 3 (CW) and 2, 4 (CCW). This is
  the spin-direction convention; flipping it would reverse yaw.
- Default integrator is Euler at 100 Hz physics rate. Low compared to
  the 500 Hz I used for the quadruped project. Worked fine here but
  may need to be re-examined when implementing MPC, where prediction
  accuracy depends on stable integration.
- Only an IMU site is defined in the model. No noise yet. For this
  project we read ground-truth state directly from MuJoCo, which is
  standard for control studies.

## On the motor mixer

- This was the only real bug in Stage 2. My first version had the pitch
  and roll torque indices swapped *and* the wrong sign convention,
  which manifested as the drone shooting in +x with motors 1, 2
  saturated and motors 3, 4 at zero. The fix was to re-derive the
  mixing matrix from the actual motor positions.
- Lesson for the next time: the signs in a quadrotor motor mixer
  depend on (a) where each motor sits relative to the body center,
  (b) which way each motor spins. Both have to be cross-checked against
  the model XML before writing the inverse mapping. Deriving it from
  memory is risky.

## On the cascaded controller architecture

- The split between outer (position PD) and inner (attitude geometric
  controller) makes debugging much easier. When hover broke, I could
  rule out the attitude loop by checking whether the motor thrusts were
  *equal* (they were, after the mixer fix), which localized the bug.
- The geometric attitude controller (Lee et al. style) works well on
  the X2 even without quaternion-aware tuning. Roll/pitch/yaw gains of
  [20, 20, 4] gave good attitude tracking; yaw is intentionally softer
  because the X2 doesn't need fast yaw response for this kind of task.
- Initial position gains [6, 6, 8] / damping [4, 4, 5] were chosen to
  be conservative. They turned out to be near-optimal for both hover
  and step. The fact that they generalized well between tasks is a
  good sign about the model's responsiveness.

## On trajectory tracking — the v_des bug

- This is the most useful diagnostic from Stage 2 and it should
  appear in the paper somewhere.
- A pure-position PD controller (with v_des implicitly = 0) cannot
  track a moving reference. On a 1.5 m radius / 1.57 m/s tangential
  circle, the controller spirals outward from the origin during the
  ramp-in, then plateaus with steady-state error roughly equal to the
  trajectory radius. The drone ends up running about a full radius
  behind the reference, which makes the actual trajectory a circle
  centered on the drone's "stuck point" rather than the reference
  center.
- Adding `v_des` (the desired velocity) brings tracking error down to
  the order of a few tens of centimeters.
- Adding `a_des_ff` (the feed-forward acceleration including
  centripetal) brings it down further to 2.2 cm RMS / 5.2 cm max on
  the same circle. This is competitive with reported drag-compensated
  results in the literature on real hardware (though our setup is
  simpler in that we have no actual drag or unmodeled dynamics here).
- The lesson: feed-forward terms matter as much as feedback gains for
  tracking moving references. MPC will get this "for free" because its
  cost function compares predicted trajectories to the full reference,
  not just instantaneous position.

## On the step test — non-minimum-phase behavior

- During a 2 m +x step, the drone briefly moves *backward* (about
  25 cm at t = 2.05 s) before accelerating forward. This is not a
  controller bug; it's the unavoidable consequence of tilt-then-
  translate dynamics in an underactuated quadrotor.
- The motors briefly saturate to 13 N during the step (motors 1, 2 at
  max while motors 3, 4 drop to ~0). A more aggressive step (say 5 m)
  would result in prolonged saturation and overshoot.
- This saturation behavior is exactly what MPC will improve on. MPC
  knows the future trajectory and the actuator limits, so it can plan
  to use the available authority budget smoothly rather than bouncing
  off the ceiling.

## On using these results for the paper

The PD baseline numbers (2.2 cm RMS / 5.2 cm max on the circle) are
the reference point. Stage 3 (MPC) should improve on these on the same
trajectory. Stage 4 (perturbations) will degrade them, and the question
becomes whether MPC degrades less, or whether the learned residual in
Stage 5 closes the additional gap.

## Things I would do differently

- Verify the motor mixing matrix on paper *before* writing the
  controller code. Saved 30 minutes of debugging.
- Build the controller assuming a moving reference from the start.
  v_des should never have defaulted silently to zero; it should have
  been required.
- Run the circle test with a coarse trajectory first (radius 0.5 m at
  low speed) to validate tracking before increasing speed. Would have
  surfaced the v_des bug earlier with a less dramatic failure.

## Status at end of Stage 2

- Hover: passes, error ~ 0 to machine precision.
- 2 m step: passes, 1.60 s settling time, brief motor saturation
  expected and acceptable.
- 1.5 m radius circle at 1.57 m/s: passes, 2.2 cm RMS error.

Stage 3 (MPC) begins next.
