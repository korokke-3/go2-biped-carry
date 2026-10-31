"""Symmetry of a trained policy on a task: mean |pi(mirror(o)) - mirror(pi(o))| over rollout observations.
Also prints the observation-term layout the mirror uses.

  python scripts/diagnostics/diag_symmetry_policy.py --task Unitree-Go2-Biped-Load-V3 --checkpoint <pt>
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
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
from go2_carry import symmetry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = 64
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
import traceback  # noqa: E402
try:
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    om = u.observation_manager
    for g in ["policy", "critic"]:
        print(g, "terms:", list(zip(om.active_terms[g], om.group_obs_term_dim[g])), "history:",
              getattr(getattr(u.cfg.observations, g), "history_length", None))
    obs = env.get_observations()
    errs = []
    with torch.inference_mode():
        for i in range(200):
            a = policy(obs)
            oa, aa = symmetry.compute_symmetric_states(env, obs=obs, actions=a)
            a_m = policy(oa[64:])
            errs.append((a_m - aa[64:]).abs().mean().item())
            obs, _, _, _ = env.step(a)
    print("slices:", symmetry._cache.get(("slices", "policy")))
    print(f"SYM_ERR mean|pi(mirror o) - mirror(pi o)| = {sum(errs) / len(errs):.4f} (first {errs[0]:.4f}, last {errs[-1]:.4f})")
except Exception:
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
env.close()
app.close()
