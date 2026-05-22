# Quadrotor Basics

A quadrotor is one of the most common aerial robots used in robotics and control research. Physically, it is a flying vehicle with four rotors (two rotating clockwise and two anti-clockwise) placed at the four corners of the body. Linear systems isn't enough to ensure their proper and smooth functioning. At first glance it looks mechanically simple, but once the dynamics and control are studied properly, it becomes a fairly complex nonlinear system.

The four propellers generate thrust forces and torques that allow the drone to move in space. If all four rotors increase speed equally, the drone starts rising upward. If the thrust becomes smaller than gravity, the drone starts descending (downward movement).

The more interesting motion (actually, the most pathetic motion) happens when rotor speeds become uneven. If one side produces more thrust than the other, the quadrotor tilts. Once tilted, the thrust vector is no longer purely vertical. A horizontal component appears, and this creates forward, backward, or sideways motion.

So a quadrotor does not directly push itself forward like a car. Instead, it changes orientation and redirects its thrust vector.

---

# Roll, Pitch, and Yaw (the most concering parameters in 3D space motion modeling)

Quadrotor orientation is usually represented using roll, pitch, and yaw angles.

- Roll → rotation about the x-axis (sideways tilting).
- Pitch → rotation about the y-axis (up and down ways tilting).
- Yaw → rotation about the z-axis (turning left or right horizontally, like changing heading direction).

These rotational motions are important because translational motion depends heavily on orientation. A quadrotor usually changes position by first changing attitude.
What is orientation: Orientation is the description of how a rigid body is rotated in 3D space relative to a reference frame (like world frame). This tells us the direction the quadrotor is facing, usually represented using roll, pitch, and yaw or a quaternion.

---

# Degrees of Freedom and Control Inputs

A quadrotor moves in 6 degrees of freedom:

1. x-position
2. y-position
3. z-position
4. roll
5. pitch
6. yaw

However, there are only four independent control inputs coming from the four rotors. Because of this, the quadrotor is considered an underactuated system.

This also means not every motion can be controlled independently at the same time. Translational and rotational motions become coupled together. For example, if the drone wants to accelerate forward, it must first pitch forward slightly. Orientation control and position control therefore become interconnected.

---

# Coordinate Frames

Quadrotor dynamics are usually described using two coordinate frames:

## 1. World Frame

The world frame (also called inertial frame) is fixed to the environment.

Position and gravity are commonly defined here.

## 2. Body Frame

The body frame moves together with the drone. Rotor thrust naturally acts in the body frame because the propellers are attached to the vehicle body. A large part of quadrotor dynamics involves transforming vectors correctly between these two frames.

---

# Basic Dynamics

The translational dynamics mainly come from Newton's Second Law:

$$
m\ddot{r} = F
$$

where:

- \(m\) = mass
- \(\ddot{r}\) = acceleration
- \(F\) = total force

This equation simply states that forces produce acceleration. The rotational dynamics come from rigid body rotational mechanics:

$$
I\dot{\omega} = \tau - \omega \times (I\omega)
$$

where:

- \(I\) = inertia matrix
- \(\omega\) = angular velocity
- \(\tau\) = applied torque

Conceptually, this equation describes how torques generate angular acceleration. Initially these equations may look intimidating, but physically they are simply describing how forces and torques affect motion.

---

# Nonlinearity in Quadrotors

Quadrotor systems are nonlinear systems.

The main reason is that translational motion depends on orientation. When the drone tilts, the thrust vector changes direction. This introduces nonlinear relationships involving trigonometric terms.

Because of this coupling, quadrotor control becomes more complicated than standard linear systems.

At small operating regions, linear approximations are often used. However, during aggressive flight or high-speed maneuvers, the full nonlinear dynamics become important.

---

# Rotor Drag and Aerodynamic Effects

In simplified quadrotor models, aerodynamic effects are sometimes ignored. At low speeds this may work reasonably well. However, when it comes to aggressive motion (high-speed), aerodynamic effects such as rotor drag start becoming significant.

Rotor drag creates additional forces opposing motion. These forces can slightly disturb the planned trajectory and reduce tracking accuracy.

Without considering drag properly, a controller may work perfectly in simulation but produce noticeable tracking errors in real-world flight.

So, it's very much understandable that this becomes especially important for:

- High-speed trajectory tracking.
- Drone racing.
- Agile aerial maneuvers.
- Autonomous navigation.

---

# Differential Flatness

One of the most important concepts in modern quadrotor control is **differential flatness**.

Initially the term sounds abstract, but the core idea is actually quite intuitive.

A system is differentially flat if all system states and control inputs can be represented using a smaller set of outputs and their derivatives.

For quadrotors, the flat outputs are commonly:

$$
\{x, y, z, \psi\}
$$

where:

- \(x,y,z\) = position
- \(\psi\) = yaw angle

From these outputs and their derivatives, it becomes possible to reconstruct:

- velocity
- acceleration
- orientation
- angular velocity
- thrust
- control inputs

This is extremely useful because it simplifies trajectory generation significantly.

Instead of solving complicated nonlinear equations directly, the process becomes:

1. Design a smooth trajectory
2. Compute derivatives
3. Recover required controls mathematically

This is why differential flatness became heavily used in quadrotor trajectory planning and aggressive autonomous flight research.

---

# Model Mismatch and Real Systems

One important issue in robotics is model mismatch. A mathematical model is always an approximation of reality.

Real systems contain:

- Aerodynamic disturbances.
- Actuator delays.
- Sensor noise.
- Flexible dynamics.
- Turbulence.
- Unmodeled nonlinearities.

As more realistic effects are added, the mathematical structure of the system can change.

In some cases:
- differential flatness may become harder to prove
- original flat outputs may stop working
- control complexity increases

There has been lots of studies around the globe for quite some times now, including the ETH Zurich Autonomous research Lab. To solve this model mismatch problems as stated above, many of the communities and research labs, combining physics-based modeling with data-driven learning instead of relying on either alone. Researchers at ETH Zurich Autonomous Systems Lab and the University of Pennsylvania GRASP Lab use ideas like differential flatness, robust control, and learning-based corrections to handle unmodeled effects such as drag and wind disturbances. At MIT Computer Science and Artificial Intelligence Laboratory, Model Predictive Control is often combined with learned residual dynamics to improve prediction accuracy in real time. At UC Berkeley Robotics and Intelligent Machines Lab, physics-informed learning is used so that neural models still obey basic physical laws. Overall, the main idea is to use simplified physics for structure and machine learning for correction. This combination allows quadrotors and other robots to remain stable and accurate even under uncertain real-world conditions.

---

# Some Last Pep Talks

Quadrotors are excellent examples of modern robotic systems because they combine multiple important areas together:

- rigid body dynamics.
- nonlinear control.
- trajectory planning.
- optimization.
- feedback systems.
- aerodynamics.
- state estimation.

Although mechanically simple, their mathematical behavior becomes highly sophisticated during autonomous flight. 

So overall, studying quadrotors provides strong intuition for understanding robotics, control systems, autonomous vehicles, and modern aerial navigation systems.