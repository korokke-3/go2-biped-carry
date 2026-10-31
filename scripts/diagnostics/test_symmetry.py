"""Check go2_carry/symmetry.py against physics: put env 1 in the mirror image of env 0's state and compare
obs(env 1) with mirror(obs(env 0)), term by term. Also checks mirror(mirror(x)) == x.

  python scripts/diagnostics/test_symmetry.py --task Unitree-Go2-Biped-GaitWalk-V5
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Unitree-Go2-Biped-GaitWalk-V5")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import go2_carry  # noqa: E402,F401
from go2_carry import mdp as bmdp  # noqa: E402
from go2_carry import symmetry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = 2
cfg.observations.policy.enable_corruption = False
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
env.reset()
robot = u.scene["robot"]
for _ in range(30):  # let it move a bit so velocities are non-zero
    u.step(torch.randn(2, u.action_manager.total_action_dim, device=u.device) * 0.5)

perm, sign = symmetry._joint_maps(u)
print("joints:", robot.joint_names)
print("mirror:", [robot.joint_names[i] for i in perm.tolist()], sign.tolist())

# mirror env 0's state into env 1
rs = robot.data.root_state_w.clone()
o = u.scene.env_origins
pos = rs[0, :3] - o[0]
q = rs[0, 3:7]  # (w, x, y, z); reflection y -> -y maps the rotation to (w, -x, y, -z)
lv, av = rs[0, 7:10], rs[0, 10:13]
rs[1, :3] = o[1] + pos * torch.tensor([1, -1, 1], device=u.device)
rs[1, 3:7] = q * torch.tensor([1, -1, 1, -1], device=u.device)
rs[1, 7:10] = lv * torch.tensor([1, -1, 1], device=u.device)
rs[1, 10:13] = av * torch.tensor([-1, 1, -1], device=u.device)
jp, jv = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
jp[1] = jp[0][perm] * sign
jv[1] = jv[0][perm] * sign
robot.write_root_state_to_sim(rs[1:2], env_ids=torch.tensor([1], device=u.device))
robot.write_joint_state_to_sim(jp[1:2], jv[1:2], env_ids=torch.tensor([1], device=u.device))
cmd = u.command_manager.get_term("base_velocity")
cmd.vel_command_b[1] = cmd.vel_command_b[0] * torch.tensor([1, -1, -1], device=u.device)
period = cfg.observations.policy.gait_phase.params["period"] if getattr(cfg.observations.policy, "gait_phase", None) else 0.6
u.episode_length_buf[1] = u.episode_length_buf[0] + int(round(0.5 * period / u.step_dt))  # half a gait period
# per-joint domain randomization (RefWalk critic terms): give env 1 the mirrored parameters of env 0
kp_all = torch.zeros(2, robot.num_joints, device=u.device); kd_all = torch.zeros_like(kp_all)
for act in robot.actuators.values():
    kp_all[:, act.joint_indices], kd_all[:, act.joint_indices] = act.stiffness, act.damping
for act in robot.actuators.values():  # env 1 gets env 0's gains, mirrored (L/R swap keeps the joint type)
    act.stiffness[1], act.damping[1] = kp_all[0][perm][act.joint_indices], kd_all[0][perm][act.joint_indices]
robot.data.joint_friction_coeff[1] = robot.data.joint_friction_coeff[0][perm]
robot.data.joint_armature[1] = robot.data.joint_armature[0][perm]
u.action_manager._action[1] = u.action_manager._action[0][perm] * sign
if bmdp.HeadingTarget.psi_des is not None:  # mirrored heading target: yaw -> -yaw
    bmdp.HeadingTarget.psi_des[1] = -bmdp.HeadingTarget.psi_des[0]
u.scene.update(0.0)
robot.update(0.0)
obs = u.observation_manager.compute(update_history=True)  # push the mirrored state as the newest frame

import traceback
ok = True
print("policy env0[:12]", obs["policy"][0, :12].tolist())
print("policy env1[:12]", obs["policy"][1, :12].tolist())
try:
  for g in ["policy", "critic"]:
    x = obs[g]
    m = symmetry._mirror_group(u, g, x[0:1])
    mm = symmetry._mirror_group(u, g, m)
    assert torch.allclose(mm, x[0:1]), "mirror is not an involution"
    for n, a, b, base in symmetry._cache[("slices", g)]:
        if n == "joint_effort":
            continue  # efforts are from the last physics step, not set by the state write
        # with observation history only the newest frame reflects the mirrored state write
        err = (x[1, b - base:b] - m[0, b - base:b]).abs().max().item()
        flag = "OK " if err < 1e-3 else "BAD"
        ok &= err < 1e-3
        print(f"{flag} {g:6s} {n:18s} max|obs(mirror state) - mirror(obs)| = {err:.2e}", flush=True)
except Exception:
  traceback.print_exc(file=sys.stdout); ok = False
print("ALL_OK" if ok else "MISMATCH")
env.close()
app.close()
