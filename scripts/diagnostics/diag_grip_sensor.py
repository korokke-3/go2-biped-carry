"""Check that the per-foot object contact sensors fire when the front feet squeeze the object."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import go2_carry  # noqa: E402,F401

cfg = load_cfg_from_registry("Unitree-Go2-Biped-Carry", "play_env_cfg_entry_point")
cfg.scene.num_envs = 4
env = gym.make("Unitree-Go2-Biped-Carry", cfg=cfg).unwrapped
env.reset()
robot = env.scene["robot"]
act_term = env.action_manager.get_term("JointPositionAction")
scale = 0.5
# hold the reset joint pose, but squeeze the front hips inwards
target = robot.data.joint_pos.clone()
for name, val in {"FL_hip_joint": -0.45, "FR_hip_joint": 0.45}.items():
    i = robot.find_joints(name)[0][0]
    target[:, i] = val
action = (target - robot.data.default_joint_pos) / scale
out = []
for k in range(25):
    env.step(action)
    fl = env.scene.sensors["fl_foot_object_contact"].data
    fr = env.scene.sensors["fr_foot_object_contact"].data
    net = env.scene.sensors["contact_forces"].data.net_forces_w
    ids = robot.find_bodies(["FL_foot", "FR_foot"], preserve_order=True)[0]
    cs = env.scene.sensors["contact_forces"]
    cids = cs.find_bodies(["FL_foot", "FR_foot"], preserve_order=True)[0]
    out.append(f"step {k}: FLobj={fl.force_matrix_w[0].norm(dim=-1).sum().item():.2f} FRobj={fr.force_matrix_w[0].norm(dim=-1).sum().item():.2f} "
               f"FLnet={net[0, cids[0]].norm().item():.2f} FRnet={net[0, cids[1]].norm().item():.2f} "
               f"fm_shape={tuple(fl.force_matrix_w.shape)} obj_z={env.scene['object'].data.root_pos_w[0, 2].item():.3f}")
with open("/tmp/diag_grip.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
env.close()
app.close()
