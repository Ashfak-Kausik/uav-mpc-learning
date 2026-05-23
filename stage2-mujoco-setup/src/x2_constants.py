"""
Physical constants for the Skydio X2 model, extracted from the
MuJoCo Menagerie scene.

All values verified empirically by inspecting the model in 02_inspect_x2.py.
"""

import os
import numpy as np

# Path to the Skydio X2 scene XML (in mujoco_menagerie).
SCENE_PATH = os.path.expanduser(
    "~/projects/mujoco_menagerie/skydio_x2/scene.xml"
)

# Mass and inertia (from model.body_mass and model.body_inertia for body "x2")
MASS = 1.325                      # kg
INERTIA_DIAG = np.array([
    0.06071129,                   # Ixx
    0.0364684,                    # Iyy
    0.0254117,                    # Izz
])
INERTIA = np.diag(INERTIA_DIAG)
INERTIA_INV = np.diag(1.0 / INERTIA_DIAG)

# Gravity
GRAVITY = 9.81                    # m/s^2

# Actuators
N_MOTORS = 4
THRUST_MIN = 0.0                  # N (per motor)
THRUST_MAX = 13.0                 # N (per motor)
HOVER_THRUST_TOTAL = MASS * GRAVITY        # ~13.0 N
HOVER_THRUST_PER_MOTOR = HOVER_THRUST_TOTAL / N_MOTORS  # ~3.25 N

# Geometry: motor positions extracted from site_pos for sites
# thrust1..thrust4. Skydio X2 is an X-configuration quadrotor.
# Reading from inspection:
#   thrust1: (-0.14, -0.18, 0.05)
#   thrust2: (-0.14,  0.18, 0.05)
#   thrust3: ( 0.14,  0.18, 0.08)
#   thrust4: ( 0.14, -0.18, 0.08)
# X-offset and Y-offset are the relevant lever arms.
ARM_X = 0.14                      # m (forward/back offset)
ARM_Y = 0.18                      # m (left/right offset)
ARM_LEN = float(np.sqrt(ARM_X**2 + ARM_Y**2))  # ~0.227 m

# Drag-to-thrust ratio (kappa), from gear[5] of the thrust actuators.
# Motors 1 and 3 have negative drag torque, motors 2 and 4 positive.
KAPPA = 0.0201

# Simulation defaults (used by some scripts as defaults; can be overridden)
DEFAULT_TIMESTEP = 0.01           # seconds (100 Hz)