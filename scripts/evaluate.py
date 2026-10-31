"""Evaluate a trained go2_carry policy: success metrics over many envs (+ optional video of env 0).

  python scripts/evaluate.py --task Unitree-Go2-Biped-Stand --checkpoint logs/.../model_1500.pt --num_envs 64 --seconds 15
  python scripts/evaluate.py ... --cmd 0.4 0 0 --video     # fixed (vx, vy, wz) command in the biped heading frame
Prints a JSON line with the metrics; video goes to <checkpoint dir>/eval_videos/.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=15.0)
p.add_argument("--cmd", type=float, nargs=3, default=None, help="fixed command (vx vy wz); default: task's own sampling")
p.add_argument("--start", choices=["mixed", "quad", "biped", "mid"], default="quad", help="initial pose for all envs")
p.add_argument("--mid_f", type=float, default=0.5, help="with --start mid: 0=quad pose, 1=upright pose")
p.add_argument("--mass", type=float, default=None, help="(carry task) fixed object mass [kg]")
p.add_argument("--load_kg", type=float, default=None, help="(load task) payload on the forearms [kg], placed suddenly")
p.add_argument("--load_on_s", type=float, default=3.0, help="(load task) time the payload is placed [s]")
p.add_argument("--load_ramp_s", type=float, default=0.0, help="(load task) seconds to ramp the payload in (0 = sudden)")
p.add_argument("--video", action="store_true")
p.add_argument("--tag", default="")
p.add_argument("--noise", action="store_true", help="keep the training observation noise")
p.add_argument("--push", action="store_true", help="keep the training push events (velocity kicks)")
p.add_argument("--obj_kg", type=float, default=None, help="(hold task) object mass [kg]; all shapes, every env gets an object")
p.add_argument("--obj_drop", type=float, default=0.0, help="(hold task) drop height at hand-over [m] (0 = gentle)")
p.add_argument("--obj_ready_s", type=float, default=1.0, help="(hold task) earliest hand-over time [s]")
p.add_argument("--obj_release_s", type=float, default=0.0,
               help="(HoldRef) gentle hand-over: hand support m g -> 0 over this time [s] (0 = put down at once)")
p.add_argument("--obj_remove_s", type=float, default=None, help="(HoldRef) take the object away this long after placement [s]")
p.add_argument("--obj_remove_ramp_s", type=float, default=0.0, help="(HoldRef) lifting time when taking it away [s]")
p.add_argument("--obj_cmd_scale", type=float, default=0.0, help="(hold task) share of --cmd given while holding (0 = stand)")
p.add_argument("--trace", action="store_true", help="save per-step traces of the first 8 envs (CoM height etc.) as .npz")
p.add_argument("--eye", type=float, nargs=3, default=(2.0, 2.0, 1.0), help="video camera offset from the robot [m]")
p.add_argument("--video_env", type=int, default=0, help="env index the video camera follows (see obj_dropped_envs)")
p.add_argument("--video_row", action="store_true",
               help="(hold task, 64 envs) film the first grid row = envs 0-7 = all 8 object shapes side by side (left: 0)")
p.add_argument("--video_group", type=int, default=None, help="with --video_row: close-up of envs g..g+3 (shapes g..g+3)")
p.add_argument("--video_row_index", type=int, default=0,
               help="with --video_row: grid row r = envs 8r..8r+7 (HoldV4 reference set: row 1 holds the 0.44 m sphere, env 11)")
p.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 0.3), help="video look-at offset from the robot [m]")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.video:
    args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import go2_carry  # noqa: E402,F401
from go2_carry import mdp as bmdp  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
if args.push:
    _train_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.events.push_robot = _train_cfg.events.push_robot
    if env_cfg.events.push_robot is not None:  # push within the 8-10 s test
        env_cfg.events.push_robot.interval_range_s = (2.0, 4.0)
env_cfg.scene.num_envs = args.num_envs
env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.seconds + 1.0)
env_cfg.observations.policy.enable_corruption = bool(args.noise)
if "biped_prob" in env_cfg.events.reset_robot.params:  # stand/walk tasks (carry always starts upright)
    env_cfg.events.reset_robot.params["biped_prob"] = {"mixed": 0.5, "quad": 0.0, "biped": 1.0, "mid": 0.0}[args.start]
    env_cfg.events.reset_robot.params["mid_prob"] = 1.0 if args.start == "mid" else 0.0
    env_cfg.events.reset_robot.params["mid_range"] = (args.mid_f, args.mid_f)
if args.cmd is not None:
    r = env_cfg.commands.base_velocity.ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = [(c, c) for c in args.cmd]
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
if args.mass is not None and hasattr(env_cfg.events, "object_mass"):
    env_cfg.events.object_mass.params["mass_distribution_params"] = (args.mass, args.mass)
if hasattr(env_cfg.events, "object_handover"):
    from go2_carry import box_mdp as hmdp
    hmdp.Box.assist = 0.0  # evaluate without the training assist
    h = env_cfg.events.object_handover.params
    if args.obj_kg is not None:
        h.update({"p_object": 1.0, "fixed_level": hmdp.DISABLED_LEVEL,  # all shapes, also the untrained one
                  "fixed_drop": args.obj_drop,
                  "fixed_ready_s": args.obj_ready_s, "obj_cmd_scale": args.obj_cmd_scale})
        if "release_range_s" in h:  # HoldRef hand-over options (fixed for evaluation)
            h.update({"fixed_release_s": args.obj_release_s, "p_remove": 0.0, "fixed_remove_s": args.obj_remove_s,
                      "remove_ramp_s": (args.obj_remove_ramp_s, args.obj_remove_ramp_s)})
        env_cfg.events.object_mass.params["fixed_mass"] = args.obj_kg
    else:
        h.update({"p_object": 0.0})
if hasattr(env_cfg.events, "forearm_load"):
    if args.load_kg is not None:
        env_cfg.events.forearm_load.params.update({"fixed_mass": args.load_kg, "fixed_on_time_s": args.load_on_s,
                                                    "fixed_ramp_s": args.load_ramp_s})
    else:
        env_cfg.events.forearm_load.params.update({"fixed_mass": 0.0, "fixed_on_time_s": 0.0})
if args.video:
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = args.video_env
    env_cfg.viewer.eye = tuple(args.eye)
    env_cfg.viewer.lookat = tuple(args.lookat)
    if args.video_row:
        # 64 envs -> 8 x 8 grid; row 0 (envs 0-7, shape i % 8 = i) sits at x = +3.5 * spacing, y = (j - 3.5) * spacing.
        # Envs do not collide with each other, so a 1.2 m spacing is fine for filming.
        sp = 1.2
        env_cfg.scene.env_spacing = sp
        x0 = (3.5 - args.video_row_index) * sp  # Isaac Lab grid: row r at x = -(r - 3.5) * spacing
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.resolution = (1920, 1080)
        if args.video_group is None:
            env_cfg.viewer.lookat = (x0, 0.0, 0.4)
            env_cfg.viewer.eye = (x0 + 7.5, 0.0, 1.8)
        else:  # close-up of envs g .. g+3 of row 0 (shapes g .. g+3), from the front, slightly above
            yc = (args.video_group + 1.5 - 3.5) * sp
            env_cfg.viewer.lookat = (x0, yc, 0.45)
            env_cfg.viewer.eye = (x0 + 3.6, yc, 1.5)

# small PhysX GPU buffers so evaluation can run next to a training job on an 8 GB GPU
env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "eval_videos")
    name = f"{os.path.basename(args.checkpoint)[:-3]}_{args.start}{'_' + args.tag if args.tag else ''}"
    env = gym.wrappers.RecordVideo(env, out_dir, step_trigger=lambda s: s == 0,
                                   video_length=int(args.seconds / env.unwrapped.step_dt), name_prefix=name,
                                   disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
robot = u.scene["robot"]
obs = env.get_observations()
steps = int(args.seconds / u.step_dt)
n = args.num_envs
alive = torch.ones(n, dtype=torch.bool, device=u.device)        # not yet fallen (first episode only)
stood = torch.zeros(n, dtype=torch.bool, device=u.device)
t_stand = torch.full((n,), float("nan"), device=u.device)
upright_steps = torch.zeros(n, device=u.device)
vel_err, yaw_err, err_count = torch.zeros(n, device=u.device), torch.zeros(n, device=u.device), torch.zeros(n, device=u.device)
yaw_rate_sum = torch.zeros(n, device=u.device)  # actual world-z yaw rate, to see under/over-turning
yaw_rate_sq = torch.zeros(n, device=u.device)  # -> per-step yaw-rate wobble around its mean (torso twist per step)
knee_ids, knee_names = robot.find_joints(["RL_calf_joint", "RR_calf_joint", "RL_thigh_joint", "RR_thigh_joint"], preserve_order=True)
q_sum = torch.zeros(len(knee_ids), device=u.device)
q_min = torch.full((len(knee_ids),), float("inf"), device=u.device)
q_max = torch.full((len(knee_ids),), -float("inf"), device=u.device)
h_sum = torch.zeros((), device=u.device)
# knee angle at hind-foot touchdown vs. deepest bend in the preceding swing (is the knee extended before landing?)
cs = u.scene.sensors["contact_forces"]
hfoot_ids, _ = cs.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)
hcalf_ids, _ = robot.find_joints(["RL_calf_joint", "RR_calf_joint"], preserve_order=True)
swing_min = torch.zeros(n, 2, device=u.device)
td_q, td_min, td_n = torch.zeros((), device=u.device), torch.zeros((), device=u.device), torch.zeros((), device=u.device)
start_xy = robot.data.root_pos_w[:, :2].clone()
rew_terms = {n: 0.0 for n in u.reward_manager.active_terms}
# posture quality while upright: sideways lean of the torso (projected gravity on body y), front-leg (arm)
# joint speed, tipping rate, and the closest the two hind feet come to each other (leg crossing)
front_ids, _ = robot.find_joints("F[L,R]_.*")
hfeet_b, _ = robot.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)
lean_s = torch.zeros(n, device=u.device); lean_sq = torch.zeros(n, device=u.device)
from go2_carry import box_mdp as _bx  # noqa: E402
tray_ids, _ = robot.find_joints(_bx.FRONT_JOINTS, preserve_order=True)
tray_q = torch.tensor(_bx._tray(), device=u.device)  # HoldV4: the waiting pose
tray_dev = torch.zeros(n, device=u.device)
arm_v = torch.zeros(n, device=u.device); tip_sq = torch.zeros(n, device=u.device); post_n = torch.zeros(n, device=u.device)
feet_dmin = torch.full((n,), float("inf"), device=u.device)
# hind knee scraping: hind thigh / calf links touching anything (in practice the ground) while upright
knee_bodies, _ = cs.find_bodies(["RL_thigh", "RR_thigh", "RL_calf", "RR_calf"], preserve_order=True)
knee_touch_steps = torch.zeros(n, device=u.device)
knee_touch_any = torch.zeros(n, dtype=torch.bool, device=u.device)
trace = {k: [] for k in ["t", "com_z", "base_z", "lean_deg", "feet_dist", "contact_L", "contact_R", "load_N"]}
masses = robot.root_physx_view.get_masses().to(u.device)  # (N, B)
v_after = torch.zeros(n, 2, device=u.device)  # mean heading-frame (fwd, left) velocity after 3 s (steady walking)
v_after_n = torch.zeros(n, device=u.device)  # per-step weighted reward per term (alive envs)
hind_ids, _ = robot.find_joints("R[L,R]_.*")
hind_names = [robot.joint_names[i] for i in hind_ids]
hind_tau_sq = torch.zeros(len(hind_ids), device=u.device)  # sum of torque^2 per hind joint while upright (heating)
tau_max = torch.zeros(len(robot.joint_names), device=u.device)
hind_tau_sum, hind_tau_n = torch.zeros(n, device=u.device), torch.zeros(n, device=u.device)
swing_min_prev = swing_min.clone()
obj_dropped_first = torch.zeros(n, dtype=torch.bool, device=u.device)
knee_term_first = torch.zeros(n, dtype=torch.bool, device=u.device)
robot_fell_first = torch.zeros(n, dtype=torch.bool, device=u.device)
obj_ever_placed = torch.zeros(n, dtype=torch.bool, device=u.device)
obj_held_steps = torch.zeros(n, device=u.device)
obj_drop_t = torch.full((n,), float("nan"), device=u.device)
# object position relative to the forearm centre (heading frame: fwd, left, up), one step before the drop
obj_rel_prev = torch.zeros(n, 3, device=u.device)
# which front-leg links touch the held object (filters of the object_contact sensor: FL_calf, FR_calf, FL_thigh, FR_thigh)
oc_calf = torch.zeros(n, device=u.device); oc_thigh = torch.zeros(n, device=u.device); oc_n = torch.zeros(n, device=u.device)
obj_rel_at_drop = torch.full((n, 3), float("nan"), device=u.device)
with torch.inference_mode():
    for i in range(steps):
        obs, _, dones, extras = env.step(policy(obs))
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
        fell = dones.bool() & ~timeouts
        if args.obj_kg is not None:  # separate: object dropped vs robot fell (first episode only)
            dr = u.termination_manager.get_term("object_dropped") & alive
            obj_dropped_first |= dr
            obj_drop_t[dr & torch.isnan(obj_drop_t)] = i * u.step_dt
            first_dr = dr & torch.isnan(obj_rel_at_drop[:, 0])
            obj_rel_at_drop[first_dr] = obj_rel_prev[first_dr]
            robot_fell_first |= (fell & ~u.termination_manager.get_term("object_dropped")) & alive
            if "knee_on_ground" in u.termination_manager.active_terms:
                knee_term_first |= u.termination_manager.get_term("knee_on_ground") & alive
            obj_held_steps += (hmdp.Box.placed & alive & ~fell).float()
            obj_ever_placed |= hmdp.Box.placed
            if "object_contact" in u.scene.sensors:
                fm = u.scene.sensors["object_contact"].data.force_matrix_w.norm(dim=-1).reshape(n, -1)  # (n, 4)
                held = (hmdp.Box.placed & alive).float()
                oc_calf += held * (fm[:, :2].max(dim=1)[0] > 1.0).float()
                oc_thigh += held * (fm[:, 2:].max(dim=1)[0] > 1.0).float()
                oc_n += held
            _c, _ = bmdp.forearm_center_w(u)
            _f, _l = bmdp._heading_frame(u)
            _d = u.scene["object"].data.root_pos_w - _c
            obj_rel_prev = torch.stack([(_d * _f).sum(1), (_d * _l).sum(1), _d[:, 2]], 1)
        alive &= ~fell
        up = (bmdp.uprightness(u) > 0.9) & (bmdp.base_height(u)[:, 0] > 0.42)
        newly = up & ~stood & alive
        t_stand[newly] = i * u.step_dt
        stood |= up & alive
        upright_steps += (up & alive).float()
        mp = (up & alive).float()
        kt = (cs.data.net_forces_w[:, knee_bodies].norm(dim=-1) > 1.0).any(dim=1) & (up & alive)
        knee_touch_steps += kt.float()
        knee_touch_any |= kt
        lean = torch.asin(robot.data.projected_gravity_b[:, 1].clamp(-1, 1))
        lean_s += mp * lean; lean_sq += mp * lean ** 2; post_n += mp
        arm_v += mp * robot.data.joint_vel[:, front_ids].abs().mean(dim=1)
        tray_dev += mp * (robot.data.joint_pos[:, tray_ids] - tray_q).abs().mean(dim=1)
        tip_sq += mp * torch.sum(robot.data.root_ang_vel_w[:, :2] ** 2, dim=1)
        fd = (robot.data.body_pos_w[:, hfeet_b[0], :2] - robot.data.body_pos_w[:, hfeet_b[1], :2]).norm(dim=1)
        feet_dmin = torch.where(mp > 0, torch.minimum(feet_dmin, fd), feet_dmin)
        if args.trace:
            k = min(8, n)
            com_z = (masses[:k] * robot.data.body_com_pos_w[:k, :, 2]).sum(1) / masses[:k].sum(1)
            cf = cs.data.net_forces_w[:k][:, hfoot_ids].norm(dim=-1) > 1.0
            ld = bmdp.ForearmLoad.force[:k].sum(1) if bmdp.ForearmLoad.force is not None else torch.zeros(k, device=u.device)
            for key, val in [("t", torch.full((k,), i * u.step_dt, device=u.device)), ("com_z", com_z),
                             ("base_z", bmdp.base_height(u)[:k, 0]), ("lean_deg", lean[:k] * 57.3), ("feet_dist", fd[:k]),
                             ("contact_L", cf[:, 0].float()), ("contact_R", cf[:, 1].float()), ("load_N", ld)]:
                trace[key].append(val.cpu().numpy())
        if i * u.step_dt >= 3.0:
            ms = (up & alive).float()
            if args.obj_kg is not None:  # carrying: only while the object is on the arms (hand-over waits)
                ms = ms * hmdp.Box.placed.float()
            v_after += ms[:, None] * bmdp.biped_lin_vel(u)
            v_after_n += ms
        for j, nme in enumerate(u.reward_manager.active_terms):
            rew_terms[nme] += (u.reward_manager._step_reward[:, j] * alive).sum().item() / max(alive.sum().item(), 1)
        tau = robot.data.applied_torque.abs()
        tau_max = torch.maximum(tau_max, (tau * alive[:, None]).max(dim=0)[0])
        hind_tau_sum += (up & alive) * tau[:, hind_ids].mean(dim=1)
        hind_tau_n += (up & alive).float()
        hind_tau_sq += ((up & alive).float()[:, None] * tau[:, hind_ids] ** 2).sum(dim=0)
        mu = up & alive  # hind joint angles while upright (knee bend check)
        if mu.any():
            q = robot.data.joint_pos[mu][:, knee_ids]
            q_sum += q.sum(dim=0); q_min = torch.minimum(q_min, q.min(dim=0)[0]); q_max = torch.maximum(q_max, q.max(dim=0)[0])
            h_sum += bmdp.base_height(u)[mu, 0].sum()
        qc = robot.data.joint_pos[:, hcalf_ids]
        in_air = cs.data.current_air_time[:, hfoot_ids] > 0
        swing_min = torch.where(in_air, torch.minimum(swing_min, qc), qc)  # reset while in contact
        td = cs.compute_first_contact(u.step_dt)[:, hfoot_ids] & (up & alive)[:, None]
        if td.any():
            td_q += qc[td].sum(); td_min += swing_min_prev[td].sum(); td_n += td.sum()
        swing_min_prev = swing_min.clone()
        if args.cmd is not None:
            m = up & alive
            cmd = u.command_manager.get_command("base_velocity")
            vel_err += m * (cmd[:, :2] - bmdp.biped_lin_vel(u)).norm(dim=1)
            yaw_err += m * (cmd[:, 2] - robot.data.root_ang_vel_w[:, 2]).abs()
            yaw_rate_sum += m * robot.data.root_ang_vel_w[:, 2]
            yaw_rate_sq += m * robot.data.root_ang_vel_w[:, 2] ** 2
            err_count += m
res = {
    "task": args.task, "ckpt": os.path.basename(args.checkpoint), "start": args.start + (f"@{args.mid_f}" if args.start == "mid" else ""), "cmd": args.cmd, "mass": args.mass, "load_kg": args.load_kg, "load_on_s": args.load_on_s if args.load_kg is not None else None,
    "envs": n, "seconds": args.seconds,
    "stood_up_rate": stood.float().mean().item(),
    "no_fall_rate": alive.float().mean().item(),
    "upright_time_frac": (upright_steps / steps).mean().item(),
    "time_to_stand_s_median": float(np.nanmedian(t_stand.cpu().numpy())) if stood.any() else None,
    "mean_horizontal_disp_m": (robot.data.root_pos_w[:, :2] - start_xy).norm(dim=1).mean().item(),
    "max_torque_Nm": {name: round(t, 1) for name, t in zip(robot.joint_names, tau_max.tolist())},
    "hind_torque_rms_upright_Nm": {nm: round(float(v), 2) for nm, v in zip(
        hind_names, (hind_tau_sq / hind_tau_n.sum().clamp(min=1)).sqrt().tolist())},  # motor heating ~ tau^2
    "hind_mean_torque_upright_Nm": (hind_tau_sum / hind_tau_n.clamp(min=1)).mean().item(),
    "hind_joint_pos_upright_rad": {nm: [round(a, 3), round(b, 3), round(c, 3)] for nm, a, b, c in zip(  # mean, min, max
        knee_names, (q_sum / hind_tau_n.sum().clamp(min=1)).tolist(), q_min.tolist(), q_max.tolist())},
    "base_height_upright_m": (h_sum / hind_tau_n.sum().clamp(min=1)).item(),
    "calf_at_touchdown_rad": (td_q / td_n.clamp(min=1)).item(),
    "calf_min_in_swing_rad": (td_min / td_n.clamp(min=1)).item(),
    "touchdowns": int(td_n.item()),
    "knee_touch_time_frac": (knee_touch_steps / post_n.clamp(min=1)).mean().item(),
    "knee_touch_env_frac": knee_touch_any.float().mean().item(),
    "lean_mean_deg": (lean_s / post_n.clamp(min=1)).mean().item() * 57.3,
    "lean_std_deg": ((lean_sq / post_n.clamp(min=1) - (lean_s / post_n.clamp(min=1)) ** 2).clamp(min=0).sqrt()).mean().item() * 57.3,
    "arm_tray_dev_rad": (tray_dev / post_n.clamp(min=1)).mean().item(),  # mean |front joint - tray pose| per joint
    "arm_joint_speed_rps": (arm_v / post_n.clamp(min=1)).mean().item(),
    "tip_rate_rms_rps": (tip_sq / post_n.clamp(min=1)).sqrt().mean().item(),
    "hind_feet_min_dist_m": feet_dmin[torch.isfinite(feet_dmin)].mean().item(),
    "vel_after3s_mps": (v_after / v_after_n.clamp(min=1)[:, None]).mean(dim=0).tolist(),
    "reward_terms_per_s": {k: round(v / steps, 4) for k, v in rew_terms.items()},
}
if args.cmd is not None:
    c = err_count.clamp(min=1)
    res["vel_err_mps"] = (vel_err / c).mean().item()
    res["yaw_err_rps"] = (yaw_err / c).mean().item()
    res["yaw_rate_mean_rps"] = (yaw_rate_sum / c).mean().item()
    res["yaw_rate_wobble_rps"] = (yaw_rate_sq / c - (yaw_rate_sum / c) ** 2).clamp(min=0).sqrt().mean().item()
if args.trace:
    tp = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "eval_traces",
                      f"{os.path.basename(args.checkpoint)[:-3]}_{args.start}{'_' + args.tag if args.tag else ''}.npz")
    os.makedirs(os.path.dirname(tp), exist_ok=True)
    np.savez(tp, cmd=np.array(args.cmd if args.cmd is not None else [np.nan] * 3), load_kg=np.nan if args.load_kg is None else args.load_kg,
             **{k: np.stack(v) for k, v in trace.items()})
    res["trace"] = tp
if args.obj_kg is not None:
    K = len(hmdp.BOX_SHAPES)
    shape = torch.arange(n, device=u.device) % K
    if hmdp.Box.ref_id is not None:  # HoldV4 fixed evaluation set: 8 old boxes (0-7) + spheres 0.30/0.36/0.44 (8-10)
        K = len(hmdp.V4_REF_BOXES) + len(hmdp.V4_REF_SPHERES)
        shape = hmdp.Box.ref_id
    res["obj_handed_over_rate"] = obj_ever_placed.float().mean().item()
    res["obj_drop_rate"] = obj_dropped_first.float().mean().item()
    res["robot_fall_rate"] = robot_fell_first.float().mean().item()  # includes knee-on-ground terminations
    res["knee_term_rate"] = knee_term_first.float().mean().item()
    res["obj_held_time_s"] = (obj_held_steps * u.step_dt).mean().item()
    res["obj_removals"] = int(hmdp.Box.removals)  # HoldRef: objects taken away (falls after that count as falls)
    res["obj_rel_at_drop"] = [[int(i)] + [round(float(v), 3) for v in obj_rel_at_drop[i]]
                              for i in obj_dropped_first.nonzero().flatten().tolist()]
    res["obj_contact_calf_frac"] = (oc_calf.sum() / oc_n.sum().clamp(min=1)).item()   # share of held time touching forearms
    res["obj_contact_thigh_frac"] = (oc_thigh.sum() / oc_n.sum().clamp(min=1)).item()  # ... touching upper arms
    if hmdp.Box.ref_id is not None:
        res["obj_contact_thigh_frac_per_ref"] = [round(float(oc_thigh[shape == j].sum() / oc_n[shape == j].sum().clamp(min=1)), 3)
                                                  for j in range(K)]
    res["obj_dropped_envs"] = [[int(i), round(float(obj_drop_t[i]), 2)] for i in obj_dropped_first.nonzero().flatten().tolist()]
    res["obj_drop_rate_per_shape"] = [round(obj_dropped_first[shape == j].float().mean().item(), 3) for j in range(K)]
    res["robot_fall_rate_per_shape"] = [round(robot_fell_first[shape == j].float().mean().item(), 3) for j in range(K)]
line = json.dumps(res)
out_jsonl = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "eval_results.jsonl")
with open(out_jsonl, "a") as f:
    f.write(line + "\n")
print("EVAL_RESULT " + line, flush=True)
sys.stderr.write("EVAL_RESULT " + line + "\n")
env.close()
app.close()
