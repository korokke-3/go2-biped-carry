"""Cross-check the Hold curriculum's drop counter against the terminations, in the TRAINING env config.

  python scripts/diagnostics/diag_hold_counter.py --checkpoint <pt> [--assist 0]
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--assist", type=float, default=0.0)
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--stochastic", action="store_true", help="sample actions like PPO does during training")
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

task = "Unitree-Go2-Biped-Hold"
cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.curriculum.object = None  # keep the state fixed
cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
try:
    hmdp.Box.assist = args.assist
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    obs = env.get_observations()
    hmdp.Box.handovers, hmdp.Box.drops = 0, 0
    term_drops = 0
    handover_ids = []
    with torch.inference_mode():
        for i in range(2250):
            a = runner.alg.policy.act(obs) if args.stochastic else policy(obs)
            obs, _, _, _ = env.step(a)
            term_drops += int(u.termination_manager.get_term("object_dropped").sum())
            if (i + 1) % 750 == 0:
                print(f"WINDOW {i + 1}: handovers={hmdp.Box.handovers} drops={hmdp.Box.drops} "
                      f"rate={hmdp.Box.drops / max(hmdp.Box.handovers, 1):.3f} present={int(hmdp.Box.present.sum())} "
                      f"placed_now={int(hmdp.Box.placed.sum())}", flush=True)
                hmdp.Box.handovers, hmdp.Box.drops = 0, 0
    print(f"COUNTER assist={args.assist} handovers={hmdp.Box.handovers} counter_drops={hmdp.Box.drops} "
          f"termination_drops={term_drops} counter_rate={hmdp.Box.drops / max(hmdp.Box.handovers, 1):.3f} "
          f"term_rate={term_drops / max(hmdp.Box.handovers, 1):.3f}")
except Exception:
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
env.close()
app.close()
