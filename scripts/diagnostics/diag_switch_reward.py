"""Per-term rewards in the quadruped mode (switch tasks): half the envs run the policy, half hold the default
quadruped pose (action 0). Start in the quadruped stance and mode, no flips. Prints the mean per-second reward of
every term over --t0..end for each group and the uprightness (2026-10-31: is the quadruped stance the better choice
under the switch rewards?)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--t0", type=float, default=2.0)
p.add_argument("--start", default="quad", choices=["quad", "biped"])
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
os.environ["SWITCH_MODE"] = "quad"
os.environ["SWITCH_AT"] = "1e9"
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import go2_carry  # noqa: E402,F401
from go2_carry import box_mdp as hmdp  # noqa: E402
from go2_carry import mdp as bmdp  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.episode_length_s = args.seconds + 1.0
r = env_cfg.events.reset_robot.params
r["biped_prob"], r["mid_prob"] = (0.0, 0.0) if args.start == "quad" else (1.0, 0.0)
env_cfg.events.object_handover.params["p_object"] = 0.0
c = env_cfg.commands.base_velocity
c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
c.rel_standing_envs = 1.0
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
hmdp.Box.assist = 0.0
u = env.unwrapped
n = args.num_envs
hold = torch.arange(n, device=u.device) >= n // 2
rm = u.reward_manager
names = rm.active_terms
acc = torch.zeros(n, len(names), device=u.device)
up = torch.zeros(n, device=u.device)
k0 = int(args.t0 / u.step_dt)
steps = int(args.seconds / u.step_dt)
obs = env.get_observations()
for k in range(steps):
    with torch.inference_mode():
        a = policy(obs)
        a[hold] = 0.0
        obs, _, _, _ = env.step(a)
    if k >= k0:
        acc += rm._step_reward
        up += bmdp.uprightness(u)
m = steps - k0
for g, sel in (("policy", ~hold), ("hold quad pose (a=0)", hold)):
    v = acc[sel].mean(0) / m
    print(f"== {g}: uprightness {up[sel].mean().item() / m:.2f}, total {v.sum().item():+.2f}/s", flush=True)
    for nm, x in sorted(zip(names, v.tolist()), key=lambda t: t[1]):
        if abs(x) > 0.005:
            print(f"   {nm:24s} {x:+.3f}", flush=True)
env.close()
app.close()
