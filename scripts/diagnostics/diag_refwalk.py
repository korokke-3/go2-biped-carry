"""Sanity checks for Unitree-Go2-Biped-RefWalk before a long run: PD gains / DR values as configured, foot height
offset (go2_quad2hand foot_height_offset 0.022), sign of the forward lean, reward terms in the upright reset pose.

  python scripts/diagnostics/diag_refwalk.py
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Unitree-Go2-Biped-RefWalk")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import go2_carry  # noqa: E402,F401
from go2_carry import mdp as bmdp  # noqa: E402
from go2_carry import refwalk_mdp as rmdp  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = 64
cfg.observations.policy.enable_corruption = False
cfg.events.push_robot = None
cfg.events.reset_robot.params.update({"biped_prob": 1.0, "mid_prob": 0.0})
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
env.reset()
robot = u.scene["robot"]
act = next(iter(robot.actuators.values()))


def rng(name, t):
    print(f"{name:28s} min {t.min().item():.4f}  max {t.max().item():.4f}  mean {t.float().mean().item():.4f}")


for nm, ac in robot.actuators.items():
    print("actuator", nm, ac.joint_names, "Y1", ac._effort_y1[0, 0].item(), "Y2", ac._effort_y2[0, 0].item(),
          "X1", ac._velocity_x1[0, 0].item(), "X2", ac._velocity_x2[0, 0].item())
rng("Kp", act.stiffness)
rng("Kd", act.damping)
rng("joint friction", robot.data.joint_friction_coeff)
rng("joint armature", robot.data.joint_armature)
rng("action scale", torch.tensor(u.action_manager.get_term("JointPositionAction")._scale))
lr = rmdp.biped_lean_roll(u)
rng("lean at upright reset [deg]", torch.rad2deg(lr[:, 0]))
rng("roll at upright reset [deg]", torch.rad2deg(lr[:, 1]))
feet = robot.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)[0]
rng("hind foot z at reset [m]", robot.data.body_pos_w[:, feet, 2])
# hold the upright pose for 1 s with PD (zero action = default quad pose, so command the upright pose)
q_up = robot.data.joint_pos.clone()
a = (q_up - robot.data.default_joint_pos) / 0.25
for i in range(50):
    u.step(a)
lr = rmdp.biped_lean_roll(u)
rng("hind foot z after 1 s [m]", robot.data.body_pos_w[:, feet, 2])
rng("base height after 1 s [m]", bmdp.base_height(u)[:, 0])
rng("lean after 1 s [deg]", torch.rad2deg(lr[:, 0]))
# sign check: tilt the head towards the belly -> lean must be > 0
belly = torch.tensor([0.0, 0.0, -1.0], device=u.device)
g = robot.data.projected_gravity_b[0]
print("projected gravity env0", g.tolist(), " (lean > 0 means gravity has a -z (belly-side... see refwalk_mdp))")
rm = u.reward_manager
for name in rm.active_terms:
    v = rm._term_cfgs[rm.active_terms.index(name)].func(u, **rm._term_cfgs[rm.active_terms.index(name)].params)
    print(f"  reward {name:24s} mean {v.float().mean().item():.4f}")
print("terminated:", u.termination_manager.terminated.float().mean().item())
env.close()
app.close()
