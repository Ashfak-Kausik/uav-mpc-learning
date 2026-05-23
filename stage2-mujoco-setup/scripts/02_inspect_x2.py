"""
Stage 2.2: Inspecting the Skydio X2 model to extract the constants we need:
    - Mass
    - Inertia (diagonal elements of the inertia tensor)
    - Arm length (distance from the center of mass to each rotor)
    - Actuator control range
    - Gravity
    - Maximum thrust per rotor (and hover thrust requirement)

    These constants feed into the controller and (later) the MPC.
"""

import os
import mujoco
import numpy as np

SCENE_PATH = os.path.expanduser("~/projects/mujoco_menagerie/skydio_x2/scene.xml")

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

print("=" * 60)
print("Skydio X2 Model Constants")
print("=" * 60)

# Drone body (find body with non-trivial mass)
print("\nBodies:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = model.body_mass[i]
    inertia = model.body_inertia[i]
    pos = model.body_pos[i]
    print(f"  {i}: {name} - mass: {mass:.4f} kg, inertia: {inertia}, position: {pos}")

# Gravity 
print(f"\nGravity vector (world): {model.opt.gravity}")
print(f"   magnitude: {np.linalg.norm(model.opt.gravity):.4f} m/s^2")

# Actuators
print("\nActuators details:")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    ctrl_range = model.actuator_ctrlrange[i]
    force_range = model.actuator_forcerange[i]
    gear = model.actuator_gear[i]
    print(f"  {i}: {name} - control range: {ctrl_range}, force range: {force_range}, gear: {gear}")

# Sites - these often mark the rotor positions
print("\nSites (potential rotor positions):")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    pos = model.site_pos[i]
    print(f"  {i}: {name} - position: {pos}")

# Simulation timestep
print(f"\nSimulation timestep (Default): {model.opt.timestep} seconds"
      f" (Frequency: {1/model.opt.timestep:.0f} Hz)")
print(f"integrator: {model.opt.integrator}")

# Quick math check: hover thrust requirement (for my understanding)
# Needs to see and find the drone body (non-world body with mass) and sum up the gravity force
drone_mass = sum(model.body_mass[i] for i in range(1, model.nbody))
g_mag = np.linalg.norm(model.opt.gravity)
hover_thrust_per_rotor = (drone_mass * g_mag) / 4
hover_total_thrust = hover_thrust_per_rotor * 4

print(f"\nHover requirements (total drone mass = {drone_mass:.4f} kg):")
print(f"  Hover thrust per rotor: {hover_thrust_per_rotor:.4f} N")
print(f"  Total hover thrust: {hover_total_thrust:.4f} N")

print(f"\n" + "=" * 60)
print("Done")