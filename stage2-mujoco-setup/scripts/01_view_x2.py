"""
Stage 2 - A quick visual check on the Skydio X2 model in MuJoCo (checking if it loads correctly).

Loads the scene, prints out the model structure information (number of bodies, joints, actuators, etc),
then opens the MuJoCo viewer with no controller. 
The drone will fall under gravity since we are commanding zero thrust.
"""

import os
import time 
import mujoco
import mujoco.viewer 

SCENE_PATH = os.path.expanduser("~/projects/mujoco_menagerie/skydio_x2/scene.xml")


print (f"Loading: {SCENE_PATH}")
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

print(f"\nModel info: ")
print(f"  nbody (rigid bodies): {model.nbody}")
print(f"  njnt (joints): {model.njnt}")
print(f"  nq (gen. coords): {model.nq}")
print(f"  nv (gen. velocities): {model.nv}")
print(f"  nu (actuators): {model.nu}")
print(f"  Actuator names:")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"    {i}: {name}")

print(f"\nInitial state: ")
print(f"  qpos: {data.qpos}")
print(f"  qvel: {data.qvel}")

print("\nOpening viewer...")
print("Quit with Ctrl+C or by closing the window.")

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(model.opt.timestep)
        
print("Viewer closed. Exiting.")
