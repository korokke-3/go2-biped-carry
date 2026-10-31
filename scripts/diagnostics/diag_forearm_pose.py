"""Where are the forearms (front calves) while a policy stands upright? Prints knee and foot positions in the
biped heading frame (forward = belly, left, up; origin = base) and the forearm slope, for placing a real box.

  python scripts/diagnostics/diag_forearm_pose.py --task Unitree-Go2-Biped-Load-V7 --checkpoint <pt> [--load_kg 2]
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--load_kg", type=float, default=0.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import go2_carry  # noqa: E402,F401
from go2_carry import mdp as bmdp  # noqa: E402

import traceback  # noqa: E402

cfg = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
cfg.scene.num_envs = 16
cfg.events.reset_robot.params.update({"biped_prob": 1.0, "mid_prob": 0.0})
r = cfg.commands.base_velocity.ranges
r.lin_vel_x = r.lin_vel_y = r.ang_vel_z = (0.0, 0.0)
cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
if hasattr(cfg.events, "forearm_load"):
    cfg.events.forearm_load.params.update({"fixed_mass": args.load_kg, "fixed_on_time_s": 0.5, "fixed_ramp_s": 0.5})
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
try:
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    robot = u.scene["robot"]
    obs = env.get_observations()
    acc = {}
    with torch.inference_mode():
        for i in range(200):
            obs, _, _, _ = env.step(policy(obs))
            if i < 100:
                continue
            h = bmdp.biped_heading_xy(u)
            fwd = torch.cat([h, torch.zeros_like(h[:, :1])], 1)
            left = torch.stack([-h[:, 1], h[:, 0], torch.zeros_like(h[:, 0])], 1)
            upv = torch.tensor([0.0, 0.0, 1.0], device=u.device).expand_as(fwd)
            for name in ["FL_calf", "FL_foot", "FR_calf", "FR_foot", "base"]:
                pw = robot.data.root_pos_w if name == "base" else robot.data.body_pos_w[:, robot.find_bodies(name)[0][0]]
                d = pw - robot.data.root_pos_w
                v = torch.stack([(d * fwd).sum(1), (d * left).sum(1), pw[:, 2] - u.scene.env_origins[:, 2]], 1)
                acc.setdefault(name, []).append(v.mean(0))
    for name, vs in acc.items():
        m = torch.stack(vs).mean(0).tolist()
        print(f"POSE {name:8s} fwd {m[0]:+.3f}  left {m[1]:+.3f}  height {m[2]:.3f} m")
    for side in ["FL", "FR"]:
        k, f = torch.stack(acc[f"{side}_calf"]).mean(0), torch.stack(acc[f"{side}_foot"]).mean(0)
        dv = f - k
        slope = torch.rad2deg(torch.atan2(dv[2], dv[:2].norm())).item()
        print(f"POSE {side} forearm knee->foot length {dv.norm().item():.3f} m, slope {slope:+.1f} deg (0 = horizontal)")
except Exception:
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
env.close()
app.close()
