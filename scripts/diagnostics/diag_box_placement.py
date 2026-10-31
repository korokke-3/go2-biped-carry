"""Box task geometry check without a policy: reset upright with the front legs in the tray pose, hold all joints
with the PD targets, hand the box over at 0.1 s and report where the forearms and the box are for 1 s.

  python scripts/diagnostics/diag_box_placement.py
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--drop", type=float, default=0.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import go2_carry  # noqa: E402,F401
from go2_carry import mdp as bmdp  # noqa: E402

cfg = load_cfg_from_registry("Unitree-Go2-Biped-Box", "play_env_cfg_entry_point")
cfg.scene.num_envs = 8
cfg.events.reset_robot.params.update({"biped_prob": 1.0, "mid_prob": 0.0, "joint_overrides": bmdp.TRAY_JOINT_POS})
cfg.events.box_handover.params.update({"p_box": 1.0, "fixed_place_s": 0.1, "fixed_drop": args.drop})
env = gym.make("Unitree-Go2-Biped-Box", cfg=cfg)
u = env.unwrapped
try:
    env.reset()
    robot = u.scene["robot"]
    q0 = robot.data.joint_pos.clone()
    act = (q0 - robot.data.default_joint_pos) / 0.5  # JointPositionAction scale 0.5, default offset
    for i in range(50):
        env.step(act)
        if i < 8:
            st = bmdp.BoxState
            print(f"DBG {i} buf={u.episode_length_buf[:8].tolist()} present={st.present[:8].int().tolist() if st.present is not None else None} "
                  f"placed={st.placed[:8].int().tolist() if st.placed is not None else None} t_place={st.t_place[:8].tolist() if st.t_place is not None else None}")
        if i in (4, 6, 10, 25, 49):
            fwd, left = bmdp._heading_frame(u)
            pts = {}
            for name in ["FL_calf", "FL_foot", "FR_calf", "FR_foot"]:
                d = robot.data.body_pos_w[:, robot.find_bodies(name)[0][0]] - robot.data.root_pos_w
                pts[name] = ((d * fwd).sum(1).mean().item(), (d * left).sum(1).mean().item(),
                             robot.data.body_pos_w[:, robot.find_bodies(name)[0][0], 2].mean().item())
            rel = bmdp.box_rel_pos(u).mean(0).tolist()
            print(f"STEP {i:2d} t={i * u.step_dt:.2f}s upright={bmdp.uprightness(u).mean().item():.2f} "
                  f"base_z={bmdp.base_height(u).mean().item():.3f} placed={bmdp.BoxState.placed.float().mean().item():.2f} "
                  f"box_rel(fwd,left,up)=({rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:+.3f}) "
                  f"on_arms={bmdp.box_on_forearms(u, 0.08).mean().item():.2f} dropped={bmdp.box_dropped(u, 0.25, 0.1).float().mean().item():.2f}")
            if i == 4:
                for k, v in pts.items():
                    print(f"  {k:8s} fwd {v[0]:+.3f} left {v[1]:+.3f} z {v[2]:.3f}")
except Exception:
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
env.close()
app.close()
