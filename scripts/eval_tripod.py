"""Evaluate a Tripod policy (go2_carry/tripod_env_cfg.py): fixed gait mode, bag and command over many envs.

  python scripts/eval_tripod.py --checkpoint logs/.../model_X.pt --mode hop --cmd 0.3 0 0
  python scripts/eval_tripod.py --checkpoint ... --mode walk --bag 2.0 --cmd 0 0 0 [--push] [--video [--video_row]]
Appends a JSON line to <checkpoint dir>/eval_results.jsonl; videos go to <checkpoint dir>/eval_videos/.
Metrics: falls (base / head / hip on the ground), bag lost, heading-frame velocity and yaw rate after 3 s, share of
time the lifted FR paw touches the ground, stance-foot lift-offs per second while standing (stepping in place),
paw-above-elbow height, hind / FL torque RMS, mean of the per-env peak stance-foot force.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--task", default="Unitree-Go2-Tripod")
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--mode", choices=["hop", "walk"], required=True)
p.add_argument("--bag", type=float, default=0.0, help="bag mass [kg] (walk mode), 0 = none")
p.add_argument("--cmd", type=float, nargs=3, default=(0.0, 0.0, 0.0))
p.add_argument("--push", action="store_true", help="training pushes (every 2-4 s)")
p.add_argument("--motor", type=float, default=None, help="fixed motor-strength scale (default: nominal 1.0)")
p.add_argument("--video", action="store_true")
p.add_argument("--video_row", action="store_true", help="fixed camera on grid row 0 (envs 0-7) instead of following env 0")
p.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import go2_carry  # noqa: E402,F401
from go2_carry import refwalk_mdp as rmdp  # noqa: E402
from go2_carry import tripod_mdp as tmdp  # noqa: E402

cfg = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = args.seconds + 1.0
r = cfg.commands.base_velocity.ranges
r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = [(c, c) for c in args.cmd]
cfg.commands.base_velocity.rel_standing_envs = 0.0
cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
cfg.commands.base_velocity.deadzone = 0.0
if args.push:
    tr = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.events.push_robot = tr.events.push_robot
    cfg.events.push_robot.interval_range_s = (2.0, 4.0)
if args.motor is not None:
    cfg.events.motor_strength = EventTerm(func=rmdp.randomize_motor_strength, mode="reset",
                                          params={"scale_range": (args.motor, args.motor)})
cfg.events.reset_tripod.params.update({"hang_range_s": (1.0, 1.0)})
tmdp.Tri.fixed_mode = {"hop": 0, "walk": 1}[args.mode]
tmdp.Tri.fixed_bag = args.bag if args.mode == "walk" else 0.0
if args.video:
    cfg.viewer.resolution = (1280, 720)
    if args.video_row:
        sp = 1.5
        cfg.scene.env_spacing = sp
        x0 = 3.5 * sp  # Isaac Lab grid: row 0 at x = 3.5 * spacing, envs 0-7 along y
        cfg.viewer.origin_type = "world"
        cfg.viewer.lookat = (x0, 0.0, 0.25)
        cfg.viewer.eye = (x0 + 6.0, -3.0, 2.0)
    else:
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (1.2, -1.8, 0.8)
        cfg.viewer.lookat = (0.0, 0.0, 0.2)
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
ck_dir = os.path.dirname(os.path.abspath(args.checkpoint))
name = f"{os.path.basename(args.checkpoint)[:-3]}_{args.mode}{'_bag%g' % args.bag if args.bag else ''}{'_' + args.tag if args.tag else ''}"
if args.video:
    env = gym.wrappers.RecordVideo(env, os.path.join(ck_dir, "eval_videos"), step_trigger=lambda s: s == 0,
                                   video_length=int(args.seconds / env.unwrapped.step_dt), name_prefix=name,
                                   disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
robot = u.scene["robot"]
cs = u.scene.sensors["contact_forces"]
n, dev = args.num_envs, u.device
free = cfg.events.reset_tripod.params.get("free_leg", "FR")  # lifted leg (TripodHind: RR); the "FR" keys below = lifted leg
stance_ids = cs.find_bodies(tmdp.stance_feet_for(free), preserve_order=True)[0]
fr_id = cs.find_bodies([f"{free}_foot"])[0]
hind_j = robot.find_joints(["RL_hip_joint", "RR_hip_joint", "RL_thigh_joint", "RR_thigh_joint", "RL_calf_joint", "RR_calf_joint"], preserve_order=True)[0]
fl_j = robot.find_joints(["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"], preserve_order=True)[0]
fr_j = robot.find_joints([f"{free}_hip_joint", f"{free}_thigh_joint", f"{free}_calf_joint"], preserve_order=True)[0]
alive = torch.ones(n, dtype=torch.bool, device=dev)
bag_ok = torch.ones(n, dtype=torch.bool, device=dev)
v_sum, v_n = torch.zeros(n, 3, device=dev), torch.zeros(n, device=dev)
fr_touch, steps_alive, liftoffs, paw_dz = torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.zeros(n, device=dev)
tq_h, tq_fl, tq_fr = torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.zeros(n, device=dev)
tq_j, q_j, pw_j = (torch.zeros(n, robot.num_joints, device=dev) for _ in range(3))
peak_f = torch.zeros(n, device=dev)
bag_z, bag_floor, bag_tr, bag_n = torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.zeros(n, device=dev)
elbow_z = torch.zeros(n, device=dev)
prev_c = None
markers = None
if args.video and tmdp.Tri.fixed_bag:  # the bag is a computed point mass: draw it (brown ball) and its strap (thin rod)
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/bag", markers={
        "bag": sim_utils.SphereCfg(radius=0.07, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.3, 0.15))),
        "strap": sim_utils.CylinderCfg(radius=0.006, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.15, 0.1)))}))
obs = env.get_observations()
steps = int(args.seconds / u.step_dt)
with torch.inference_mode():
    for k in range(steps):
        act = policy(obs)
        obs, _, dones, _ = env.step(act)
        t = (k + 1) * u.step_dt
        if markers is not None:
            from isaaclab.utils.math import quat_from_matrix
            T = tmdp.Tri
            el = tmdp._elbow(u)
            hide = (~T.hung)[:, None].float() * torch.tensor([0.0, 0.0, -10.0], device=dev)
            d = T.pos - el
            ln = d.norm(dim=1).clamp(min=1e-4)
            z = d / ln[:, None]  # strap axis (cylinder z) from the elbow to the bag
            x = torch.nn.functional.normalize(torch.cross(z, torch.tensor([[0.0, 1.0, 0.0]], device=dev).expand(n, 3), dim=1), dim=1)
            y = torch.cross(z, x, dim=1)
            qs = quat_from_matrix(torch.stack([x, y, z], dim=2))
            pos = torch.cat([T.pos + hide, (el + T.pos) / 2 + hide])
            ori = torch.cat([torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(n, 4), qs])
            sc = torch.cat([torch.ones(n, 3, device=dev), torch.stack([torch.ones(n, device=dev), torch.ones(n, device=dev), ln], 1)])
            markers.visualize(translations=pos, orientations=ori, scales=sc,
                              marker_indices=torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)]).to(dev))
        tm = u.termination_manager
        fell = tm.get_term("body_contact") & alive
        lost = tm.get_term("bag_lost") & alive & bag_ok
        alive &= ~fell  # falls only (a lost bag also ends the episode; that env keeps no bag afterwards)
        bag_ok &= ~lost
        a = alive.float()
        f = cs.data.net_forces_w[:, stance_ids].norm(dim=-1)
        c = f > 1.0
        if prev_c is not None and t > 3.0:
            liftoffs += ((prev_c & ~c).sum(dim=1).float()) * a
        prev_c = c
        peak_f = torch.maximum(peak_f, f.max(dim=1)[0] * a)
        fr_touch += (cs.data.net_forces_w[:, fr_id].norm(dim=-1).squeeze(1) > 1.0).float() * a
        steps_alive += a
        paw_dz += (tmdp._paw(u)[:, 2] - tmdp._elbow(u)[:, 2]) * a
        elbow_z += (tmdp._elbow(u)[:, 2] - u.scene.env_origins[:, 2]) * a
        hb = tmdp.Tri.hung.float() * a
        bz = tmdp.Tri.pos[:, 2] - u.scene.env_origins[:, 2]
        bag_z += bz * hb
        bag_floor += (bz < 0.045).float() * hb
        bag_tr += tmdp.Tri.tension.norm(dim=1) / (tmdp.Tri.mass * 9.81).clamp(min=1e-3) * hb
        bag_n += hb
        tau = robot.data.applied_torque
        tq_h += tau[:, hind_j].square().mean(dim=1) * a
        tq_fl += tau[:, fl_j].square().mean(dim=1) * a
        tq_fr += tau[:, fr_j].square().mean(dim=1) * a
        tq_j += tau.square() * a[:, None]
        q_j += robot.data.joint_pos * a[:, None]
        pw_j += (tau * robot.data.joint_vel).abs() * a[:, None]
        if t > 3.0:
            yaw = torch.atan2(2 * (robot.data.root_quat_w[:, 0] * robot.data.root_quat_w[:, 3] + robot.data.root_quat_w[:, 1] * robot.data.root_quat_w[:, 2]),
                              1 - 2 * (robot.data.root_quat_w[:, 2] ** 2 + robot.data.root_quat_w[:, 3] ** 2))
            vw = robot.data.root_lin_vel_w
            vx = vw[:, 0] * torch.cos(yaw) + vw[:, 1] * torch.sin(yaw)
            vy = -vw[:, 0] * torch.sin(yaw) + vw[:, 1] * torch.cos(yaw)
            v_sum += torch.stack([vx, vy, robot.data.root_ang_vel_w[:, 2]], 1) * a[:, None]
            v_n += a
sa = steps_alive.clamp(min=1)
vn = v_n.clamp(min=1)
moving_s = max(args.seconds - 3.0, 1e-6)
res = {
    "task": args.task, "free_leg": free, "ckpt": os.path.basename(args.checkpoint), "tag": args.tag, "mode": args.mode, "bag_kg": args.bag,
    "cmd": list(args.cmd), "envs": n, "seconds": args.seconds, "push": args.push, "motor": args.motor,
    "fall_rate": float((~alive).float().mean()),
    "bag_lost_rate": float((~bag_ok).float().mean()) if args.bag > 0 else None,
    "vel_after3s": [round(float(x), 3) for x in (v_sum / vn[:, None]).mean(dim=0)],
    "fr_paw_touch_frac": float((fr_touch / sa).mean()),
    "stance_liftoffs_per_s": float((liftoffs / moving_s).mean()),
    "paw_above_elbow_m": float((paw_dz / sa).mean()),
    "torque_rms_Nm": {"hind": round(float((tq_h / sa).mean().sqrt()), 2), "FL": round(float((tq_fl / sa).mean().sqrt()), 2),
                      "FR": round(float((tq_fr / sa).mean().sqrt()), 2)},
    "joint_torque_rms_Nm": {nm: round(float(v), 2) for nm, v in zip(robot.joint_names, (tq_j / sa[:, None]).mean(dim=0).sqrt())},
    "joint_pos_mean_rad": {nm: round(float(v), 2) for nm, v in zip(robot.joint_names, (q_j / sa[:, None]).mean(dim=0))},
    "joint_abs_power_W": {nm: round(float(v), 1) for nm, v in zip(robot.joint_names, (pw_j / sa[:, None]).mean(dim=0))},
    "peak_foot_force_N": round(float(peak_f.mean()), 1),
    "elbow_height_m": round(float((elbow_z / sa).mean()), 3),
    "bag_height_m": round(float((bag_z / bag_n.clamp(min=1)).mean()), 3) if args.bag > 0 else None,
    "bag_on_floor_frac": round(float((bag_floor / bag_n.clamp(min=1)).mean()), 3) if args.bag > 0 else None,
    "bag_tension_over_mg": round(float((bag_tr / bag_n.clamp(min=1)).mean()), 3) if args.bag > 0 else None,
}
print(json.dumps(res))
with open(os.path.join(ck_dir, "eval_results.jsonl"), "a") as fh:
    fh.write(json.dumps(res) + "\n")
env.close()
app.close()
