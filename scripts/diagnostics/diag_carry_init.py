"""Print front-foot and object positions (robot base frame) right after a carry-task reset."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import go2_carry  # noqa: E402,F401

cfg = load_cfg_from_registry("Unitree-Go2-Biped-Carry", "play_env_cfg_entry_point")
cfg.scene.num_envs = 4
env = gym.make("Unitree-Go2-Biped-Carry", cfg=cfg).unwrapped
env.reset()
robot, obj = env.scene["robot"], env.scene["object"]
ids, names = robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
for k in range(3):
    q, p = robot.data.root_quat_w, robot.data.root_pos_w
    feet_b = [quat_apply_inverse(q, robot.data.body_pos_w[:, i] - p)[0].tolist() for i in ids]
    obj_b = quat_apply_inverse(q, obj.data.root_pos_w - p)[0].tolist()
    lines = [f"step {k}: base_z={p[0, 2]:.3f} up={-robot.data.projected_gravity_b[0, 0]:.2f} obj_b={[round(x, 3) for x in obj_b]}"]
    lines += [f"   {n}_b={[round(x, 3) for x in f]}" for n, f in zip(names, feet_b)]
    with open("/tmp/diag_carry.txt", "a") as fh:
        fh.write("\n".join(lines) + "\n")
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
env.close()
app.close()
