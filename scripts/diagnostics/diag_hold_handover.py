"""Step-by-step look at a hand-over under a policy: object velocity, contact force with the front legs, front-leg
deviation from the tray pose, object height above the forearms. One env per shape.

  python scripts/diagnostics/diag_hold_handover.py --checkpoint <pt> [--mass 0.3]
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--mass", type=float, default=0.3)
p.add_argument("--task", default="Unitree-Go2-Biped-Hold")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import go2_carry  # noqa: E402,F401
from go2_carry import box_mdp as hmdp  # noqa: E402
from go2_carry import mdp as bmdp  # noqa: E402

K = len(hmdp.BOX_SHAPES)
hmdp.Box.assist = 0.0
cfg = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
cfg.scene.num_envs = K
cfg.events.reset_robot.params.update({"biped_prob": 1.0, "mid_prob": 0.0})
cfg.events.object_handover.params.update({"p_object": 1.0, "fixed_level": hmdp.MAX_LEVEL, "fixed_ready_s": 1.0,
                                           "fixed_drop": 0.0})
cfg.events.object_mass.params["fixed_mass"] = args.mass
r = cfg.commands.base_velocity.ranges
r.lin_vel_x = r.lin_vel_y = r.ang_vel_z = (0.0, 0.0)
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
try:
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    obj = u.scene["object"]
    sens = u.scene.sensors["object_contact"]
    obs = env.get_observations()
    t_place = [None] * K
    with torch.inference_mode():
        for i in range(400):
            obs, _, _, _ = env.step(policy(obs))
            placed = hmdp.Box.placed
            dev = hmdp._front_dev(u)
            c, top = bmdp.forearm_center_w(u)
            for j in range(K):
                if placed[j] and t_place[j] is None:
                    t_place[j] = i
                if t_place[j] is not None and (i - t_place[j]) % 10 == 0 and i - t_place[j] <= 250 and j == 1:
                    f = sens.data.force_matrix_w[j].norm(dim=-1).reshape(-1)
                    v = obj.data.root_lin_vel_w[j]
                    w = obj.data.root_ang_vel_w[j]
                    print(f"shape{j} +{i - t_place[j]:2d} placed={bool(placed[j])} dev_tray={dev[j]:.2f} "
                          f"obj_z-top={obj.data.root_pos_w[j, 2] - top[j]:+.3f} v=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) "
                          f"|w|={w.norm():.2f} F[FLc,FRc,FLt,FRt]={[round(x, 1) for x in f.tolist()]} "
                          f"W={hmdp.bmdp.object_mass(u)[j, 0].item() * 9.81:.1f}N")
            if i == 40:
                print("dev_tray at 0.8s per shape:", [round(x, 2) for x in dev.tolist()])
except Exception:
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
env.close()
app.close()
