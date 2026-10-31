"""Export an RSL-RL RefWalk / HoldRef actor to ONNX + a deploy config (JSON) for sim2sim (deploy/sim2sim_mujoco.py)
and, later, the real robot.

  python scripts/export_onnx.py logs/rsl_rl/.../model_5999.pt   # -> <run>/exported/policy.onnx, deploy.json

The actor input is Isaac Lab's flattened history: for each term (in order), HISTORY frames oldest -> newest.
Joint order of the policy (Isaac Lab, grouped by joint type) is written to deploy.json, so the deploy side maps
joints by NAME (MuJoCo: per leg FL, FR, RL, RR; Unitree SDK: FR, FL, RR, RL).
"""
import argparse
import json
import os

import torch

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--out", default=None)
p.add_argument("--tripod", choices=["hop", "walk"], default=None, help="Tripod task: gait mode to deploy")
p.add_argument("--switch", action="store_true", help="biped <-> quadruped switching task (switch_env_cfg.py)")
a = p.parse_args()

ISAAC_JOINTS = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
# unitree_rl_lab UNITREE_GO2_CFG init_state.joint_pos (= action offset, use_default_offset=True)
DEFAULT = {"hip": {"L": 0.1, "R": -0.1}, "thigh": {"F": 0.8, "R": 1.0}, "calf": -1.5}


def default_angle(n):
    leg, kind = n[:2], n.split("_")[1]
    if kind == "hip":
        return DEFAULT["hip"][leg[1]]
    if kind == "thigh":
        return DEFAULT["thigh"][leg[0]]
    return DEFAULT["calf"]


# RefWalk (go2_carry/refwalk_env_cfg.py)
cfg = {
    "joint_names": ISAAC_JOINTS,
    "default_angles": [default_angle(n) for n in ISAAC_JOINTS],
    "kp": 30.0, "kd": 0.75, "action_scale": 0.25,
    "policy_dt": 0.02, "history": 5, "gait_period": 0.4,
    "obs_terms": [["base_ang_vel", 3, 0.25], ["projected_gravity", 3, 1.0], ["velocity_commands", 3, [2.0, 2.0, 0.25]],
                  ["joint_pos_rel", 12, 1.0], ["joint_vel_rel", 12, 0.05], ["last_action", 12, 1.0],
                  ["gait_phase", 2, 1.0]],
    "cmd_range": {"vx": [-0.4, 0.4], "vy": [0.0, 0.0], "wz": [-0.5, 0.5]},
}
if a.tripod is not None:  # Tripod (go2_carry/tripod_env_cfg.py): + gait mode one-hot, clock with the mode's period
    mode = {"hop": 0, "walk": 1}[a.tripod]
    cfg["obs_terms"] = cfg["obs_terms"][:-1] + [["gait_mode", 2, 1.0], ["gait_phase", 2, 1.0]]
    cfg["gait_mode"] = mode
    cfg["gait_period"] = [0.4, 0.6][mode]  # tripod_mdp.GAIT
    cfg["cmd_range"] = {"vx": [-0.3, 0.5], "vy": [-0.2, 0.2], "wz": [-0.6, 0.6]}
    # both modes (index into the one-hot, clock period) so the robot side can switch between them (deploy/policy.py)
    cfg["gait_modes"] = {"hop": [0, 0.4], "walk": [1, 0.6]}
if a.switch:  # switch_env_cfg: + [quad, biped] one-hot after the clock; the clock keeps the biped period
    cfg["obs_terms"] = cfg["obs_terms"] + [["gait_mode", 2, 5.0]]  # switch_env_cfg.MODE_OBS_SCALE
    cfg["gait_mode"] = 1
    cfg["gait_modes"] = {"quad": [0, 0.4], "biped": [1, 0.4]}
    cfg["cmd_range"] = {"vx": [-0.8, 1.2], "vy": [-0.5, 0.5], "wz": [-1.5, 1.5]}
cfg["cmd_deadzone"] = 0.1  # refwalk_mdp.DeadzoneVelocityCommandCfg: each component below 0.1 is trained as 0
n_in = sum(t[1] for t in cfg["obs_terms"]) * cfg["history"]

sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
layers = []
i = 0
while f"actor.{i}.weight" in sd:
    w = sd[f"actor.{i}.weight"]
    lin = torch.nn.Linear(w.shape[1], w.shape[0])
    lin.weight.data, lin.bias.data = w.clone(), sd[f"actor.{i}.bias"].clone()
    layers += [lin, torch.nn.ELU()]
    i += 2
actor = torch.nn.Sequential(*layers[:-1]).eval()
assert actor[0].in_features == n_in, (actor[0].in_features, n_in)

out = a.out or os.path.join(os.path.dirname(a.checkpoint), "exported" + (f"_{a.tripod}" if a.tripod else ""))
# the tripod / switch one-hots must not be confused: "gait_mode" index 1 = walk (Tripod) / biped (switch)
os.makedirs(out, exist_ok=True)
x = torch.zeros(1, n_in)
torch.onnx.export(actor, x, os.path.join(out, "policy.onnx"), input_names=["obs"], output_names=["actions"],
                  opset_version=17, dynamo=False, dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}})
cfg["checkpoint"] = os.path.abspath(a.checkpoint)
with open(os.path.join(out, "deploy.json"), "w") as f:
    json.dump(cfg, f, indent=1)
# reference output for a parity check on the deploy side
torch.manual_seed(0)
xt = torch.randn(4, n_in)
torch.save({"x": xt, "y": actor(xt).detach()}, os.path.join(out, "parity.pt"))
with open(os.path.join(out, "parity.json"), "w") as f:
    json.dump({"x": xt.tolist(), "y": actor(xt).detach().tolist()}, f)
print("exported", out, "inputs", n_in)
