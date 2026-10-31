"""Tripod sanity check without a policy: hold the default stance with FR in LIFT_POSE (PD targets), hang a bag and
print the paw-above-elbow height, the rope tension vs m g and the bag height.

  python scripts/diagnostics/diag_tripod.py [--bag 2.0] [--seconds 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--bag", type=float, default=2.0)
p.add_argument("--seconds", type=float, default=4.0)
p.add_argument("--task", default="Unitree-Go2-Tripod")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import go2_carry  # noqa: E402,F401
from go2_carry import tripod_mdp as tmdp  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, num_envs=16)
cfg.events.push_robot = None
cfg.events.motor_strength = None
cfg.events.reset_tripod.params.update({"p_lifted": 1.0, "hang_range_s": (0.5, 0.5)})
cfg.terminations.bag_lost = None
cfg.terminations.body_contact = None
cfg.observations.policy.enable_corruption = False
cfg.scene.robot.spawn.articulation_props.fix_root_link = True  # body held in the air: pose geometry + free bag
cfg.events.reset_robot = None
tmdp.Tri.fixed_mode, tmdp.Tri.fixed_bag = 1, args.bag
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
env.reset()
free = cfg.events.reset_tripod.params.get("free_leg", "FR")
lift = cfg.events.reset_tripod.params.get("lift_pose") or tmdp.LIFT_POSE
jid = robot.find_joints([f"{free}_hip_joint", f"{free}_thigh_joint", f"{free}_calf_joint"], preserve_order=True)[0]
act = torch.zeros(u.num_envs, robot.num_joints, device=u.device)
act[:, jid] = (torch.tensor(lift, device=u.device) - robot.data.default_joint_pos[0, jid]) / 0.25
u.command_manager.get_term("base_velocity").vel_command_b[:] = 0.0
for k in range(int(args.seconds / u.step_dt)):
    env.step(act)
    u.command_manager.get_term("base_velocity").vel_command_b[:] = 0.0
    if k % 25 == 24:
        el, pw = tmdp._elbow(u), tmdp._paw(u)
        T = tmdp.Tri
        print(f"t {u.episode_length_buf[0].item() * u.step_dt:4.1f}  paw-elbow dz {float((pw - el)[:, 2].mean()):+.3f} m"
              f"  elbow z {float(el[:, 2].mean() - u.scene.env_origins[:, 2].mean()):.3f}  hung {T.hung.float().mean():.2f}"
              f"  |T| {float(T.tension.norm(dim=1).mean()):.1f} N (m g {args.bag * 9.81:.1f})"
              f"  bag z {float((T.pos[:, 2] - u.scene.env_origins[:, 2]).mean()):.3f}"
              f"  base h {float((robot.data.root_pos_w[:, 2] - u.scene.env_origins[:, 2]).mean()):.3f}"
              f"  upright {float(-robot.data.projected_gravity_b[:, 2].mean()):.2f}", flush=True)
        hip = robot.data.body_pos_w[:, robot.find_bodies(f"{free}_thigh")[0][0]]
        from isaaclab.utils.math import quat_apply_inverse
        q = robot.data.root_quat_w
        print(f"    base frame (x fwd, z up) rel. {free} hip: knee", [round(v, 3) for v in quat_apply_inverse(q, el - hip)[0].tolist()],
              " paw", [round(v, 3) for v in quat_apply_inverse(q, pw - hip)[0].tolist()],
              " bag", [round(v, 3) for v in quat_apply_inverse(q, T.pos - hip)[0].tolist()], flush=True)
T = tmdp.Tri
el = tmdp._elbow(u)
print("per env |T|", [round(v, 1) for v in T.tension.norm(dim=1).tolist()])
print("per env rope", [round(v, 3) for v in T.rope.tolist()])
print("per env dist", [round(v, 3) for v in (T.pos - el).norm(dim=1).tolist()])
print("per env bag z", [round(v, 3) for v in (T.pos[:, 2] - u.scene.env_origins[:, 2]).tolist()])
env.close()
app.close()
