"""Hold task check without a policy: upright, front legs PD-held in the tray pose (or folded), one env per shape,
hand-over at 0.1 s; prints per shape where the object is after 0.5 s / 1.5 s and whether it was dropped.

  python scripts/diagnostics/diag_hold_placement.py [--fold] [--video]
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--fold", action="store_true", help="hold the forearms folded (calf -2.2) instead of the tray pose")
p.add_argument("--mass", type=float, default=1.0)
p.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import go2_carry  # noqa: E402,F401
from go2_carry import box_mdp as hmdp  # noqa: E402
from go2_carry import mdp as bmdp  # noqa: E402

K = len(hmdp.BOX_SHAPES)
cfg = load_cfg_from_registry("Unitree-Go2-Biped-Hold", "play_env_cfg_entry_point")
cfg.scene.num_envs = K
tray = {"F[L,R]_hip_joint": 0.0, "F[L,R]_thigh_joint": 1.3, "F[L,R]_calf_joint": -2.2 if args.fold else -1.3}
cfg.events.reset_robot.params.update({"biped_prob": 1.0, "mid_prob": 0.0, "joint_overrides": tray})
cfg.events.object_handover.params.update({"p_object": 1.0, "fixed_level": hmdp.MAX_LEVEL, "fixed_ready_s": 0.1,
                                           "fixed_drop": 0.0, "max_wait_s": 0.0})
cfg.events.object_mass.params["fixed_mass"] = args.mass
cfg.events.push_robot = None
if args.video:
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (1.2, 1.2, 0.3)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
env = gym.make("Unitree-Go2-Biped-Hold", cfg=cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    env = gym.wrappers.RecordVideo(env, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "diag_videos"),
                                   step_trigger=lambda s: s == 0, video_length=75,
                                   name_prefix="hold_placement" + ("_fold" if args.fold else ""), disable_logger=True)
u = env.unwrapped
try:
    env.reset()
    robot = u.scene["robot"]
    q0 = robot.data.joint_pos.clone()
    act = (q0 - robot.data.default_joint_pos) / 0.5
    dropped = torch.zeros(K, dtype=torch.bool, device=u.device)
    fallen = torch.zeros(K, dtype=torch.bool, device=u.device)
    for i in range(75):
        _, _, term, trunc, _ = env.step(act)
        dropped |= u.termination_manager.get_term("object_dropped")
        fallen |= u.termination_manager.get_term("body_contact")
        if i in (25, 74):
            rel = hmdp.obj_rel_pos(u)
            fell = (~hmdp.Box.placed).tolist()
            for j in range(K):
                d, k, c, lv = hmdp.BOX_SHAPES[j]
                print(f"T{i * u.step_dt:.1f}s shape{j} {d} k={k:.0f} placed={bool(hmdp.Box.placed[j])} "
                      f"rel(fwd,left,up)=({rel[j, 0]:+.3f},{rel[j, 1]:+.3f},{rel[j, 2]:+.3f}) "
                      f"DROPPED={bool(dropped[j])} robot_fell={bool(fallen[j])} upright={bmdp.uprightness(u)[j]:.2f}")
except Exception:
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
env.close()
app.close()
