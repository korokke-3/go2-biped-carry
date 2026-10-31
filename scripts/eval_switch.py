"""Evaluate biped <-> quadruped switching (switch_env_cfg.py tasks), no objects.

  python scripts/eval_switch.py --task Unitree-Go2-Biped-HoldBall-Switch --checkpoint <pt> --test down|up|cycle

  down : start upright in the biped mode, switch to quadruped at --at s
  up   : start in the quadruped stance and mode, switch to biped at --at s
  cycle: start upright, the training schedule flips the mode every U(3, 8) s, --seconds long
Measured joints at their nominal values (play cfg). Per test: falls (a termination other than time-out, first
episode only), time from the switch to the target posture (quad: body level within 15 deg, base 0.25-0.45 m, four feet
down; biped: uprightness > 0.9), and at the end: the largest joint deviation from the default quadruped pose (the robot
controller lies down / stands up from there, deploy/go2_real.py: A only within 0.2 rad), the RMS torque of the hottest
motor while in the target mode, flips per env. Appends one JSON line to <checkpoint dir>/eval_switch.jsonl.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--test", choices=["down", "up", "cycle"], required=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--at", type=float, default=3.0)
p.add_argument("--push", action="store_true", help="keep the training pushes")
p.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
os.environ["SWITCH_MODE"] = {"down": "biped", "up": "quad", "cycle": "biped"}[args.test]
if args.test != "cycle":
    os.environ["SWITCH_AT"] = str(args.at)
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import go2_carry  # noqa: E402,F401
from go2_carry import box_mdp as hmdp  # noqa: E402
from go2_carry import mdp as bmdp  # noqa: E402
from go2_carry import switch_mdp as smdp  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
train_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.episode_length_s = args.seconds + 1.0
r = env_cfg.events.reset_robot.params
r["biped_prob"], r["mid_prob"] = (0.0, 0.0) if args.test == "up" else (1.0, 0.0)
env_cfg.events.object_handover.params["p_object"] = 0.0
c = env_cfg.commands.base_velocity
c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
c.rel_standing_envs = 1.0
if args.push:
    env_cfg.events.push_robot = train_cfg.events.push_robot
env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
hmdp.Box.assist = 0.0

u = env.unwrapped
robot = u.scene["robot"]
dev, n = u.device, args.num_envs
cs = u.scene.sensors["contact_forces"]
feet, _ = cs.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
alive = torch.ones(n, dtype=torch.bool, device=dev)
t_switch = torch.full((n,), float("nan"), device=dev)
t_done = torch.full((n,), float("nan"), device=dev)
tau_sq = torch.zeros(len(robot.joint_names), device=dev)
tau_n = torch.zeros((), device=dev)
prev_quad = smdp.Mode.quad.clone() if smdp.Mode.quad is not None else None
flips = torch.zeros(n, device=dev)
quad_ok_time = torch.zeros(n, device=dev)
quad_time = torch.zeros(n, device=dev)
obs = env.get_observations()
steps = int(args.seconds / u.step_dt)
for k in range(steps):
    with torch.inference_mode():
        obs, _, dones, extras = env.step(policy(obs))
    t = (k + 1) * u.step_dt
    term = u.termination_manager.terminated & ~u.termination_manager.time_outs
    alive &= ~term
    alive &= ~(dones.bool() & ~u.termination_manager.time_outs)  # (first episode only)
    # base / head on the ground no longer ends quadruped-mode episodes (switch_env_cfg): still a fall here
    alive &= ~bmdp.body_touches_ground(u, ["base_ground", "head_upper_ground", "head_lower_ground"])
    q = smdp.Mode.quad.clone()
    if prev_quad is not None:
        changed = (q != prev_quad) & alive & (t > 0.1)  # the reset step sets the start mode
        flips += changed.float()
        t_switch = torch.where(changed & torch.isnan(t_switch), torch.full_like(t_switch, t), t_switch)
    prev_quad = q
    g = robot.data.projected_gravity_b
    h = robot.data.root_pos_w[:, 2] - u.scene.env_origins[:, 2]
    down = (cs.data.net_forces_w[:, feet, :].norm(dim=-1) > 1.0).all(dim=1)
    quad_ok = (g[:, 2] < -0.966) & (h > 0.25) & (h < 0.45) & down
    biped_ok = bmdp.uprightness(u) > 0.9
    ok = torch.where(q, quad_ok, biped_ok)
    reached = alive & ~torch.isnan(t_switch) & torch.isnan(t_done) & ok
    t_done = torch.where(reached, torch.full_like(t_done, t), t_done)
    quad_time += (q & alive).float()
    quad_ok_time += (q & alive & quad_ok).float()
    if args.test != "cycle":
        in_target = alive & ~torch.isnan(t_done)
        tau = robot.data.applied_torque[in_target]
        tau_sq += (tau ** 2).sum(0)
        tau_n += in_target.sum()

dev_end = (robot.data.joint_pos - robot.data.default_joint_pos).abs().max(dim=1).values
sw = ~torch.isnan(t_switch)
dt_done = (t_done - t_switch)[sw & ~torch.isnan(t_done)]
res = {"checkpoint": args.checkpoint, "task": args.task, "test": args.test, "tag": args.tag, "at": args.at,
       "push": args.push, "seconds": args.seconds, "num_envs": n,
       "fall_rate": round(1.0 - alive.float().mean().item(), 3),
       "switched_frac": round(sw.float().mean().item(), 3),
       "reached_frac": round((~torch.isnan(t_done) & alive).float().mean().item(), 3)}
if len(dt_done):
    res["time_to_target_s"] = {"median": round(dt_done.median().item(), 2), "p90": round(dt_done.quantile(0.9).item(), 2)}
if args.test == "down":
    res["end_joint_dev_rad"] = {"median": round(dev_end[alive].median().item(), 3),
                                "max": round(dev_end[alive].max().item(), 3),
                                "frac_below_0.2": round((dev_end[alive] < 0.2).float().mean().item(), 3)}
if args.test != "cycle" and tau_n > 0:
    rms = (tau_sq / tau_n).sqrt()
    i = int(rms.argmax())
    res["target_mode_torque_rms_Nm"] = {"max_joint": robot.joint_names[i], "max": round(rms[i].item(), 1),
                                        "per_joint": {nm: round(v, 1) for nm, v in zip(robot.joint_names, rms.tolist())}}
if args.test == "cycle":
    res["flips_per_env"] = round(flips.mean().item(), 2)
    res["quad_time_in_quad_posture"] = round((quad_ok_time.sum() / quad_time.sum().clamp(min=1)).item(), 3)
print(json.dumps(res))
with open(os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "eval_switch.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
env.close()
app.close()
