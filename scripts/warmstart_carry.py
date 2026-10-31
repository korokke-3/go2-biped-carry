"""Warm-start the carry policy from a trained walk policy.

The carry task adds (a) a 3-step proprioceptive history to the actor input and (b) object
state to the critic input. We copy the walk weights into the matching input columns
(newest history frame / original critic terms) and zero the new columns, so the warm-started
policy initially behaves exactly like the walk policy.

  python scripts/warmstart_carry.py <walk model.pt> <out model.pt> [critic_extra]
  (critic_extra: number of critic inputs appended by the new task; carry = 7, load = 2)
"""
import sys

import torch

# policy terms (dim) in order; carry uses history_length=3, flattened per term, oldest -> newest
POLICY_TERMS = [3, 3, 3, 12, 12, 12]  # ang_vel, gravity, commands, joint_pos, joint_vel, last_action
HIST = 3
CRITIC_OLD = 63  # see biped_env_cfg.ObservationsCfg.CriticCfg
CRITIC_NEW_EXTRA = 3 + 3 + 1  # object_pos_b, object_rel_vel_b, object_mass

src, dst = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    CRITIC_NEW_EXTRA = int(sys.argv[3])
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]

keys = list(sd.keys())  # parameter order == optimizer param order (all entries are parameters)


def widen_actor(w):
    """(rows, 45) -> (rows, 135): copy into the newest history frame of each term, zeros elsewhere."""
    out = torch.zeros(w.shape[0], sum(POLICY_TERMS) * HIST, dtype=w.dtype)
    o_old, o_new = 0, 0
    for d in POLICY_TERMS:
        newest = o_new + (HIST - 1) * d
        out[:, newest:newest + d] = w[:, o_old:o_old + d]
        o_old += d
        o_new += HIST * d
    return out


def widen_critic(w):
    assert w.shape[1] == CRITIC_OLD, w.shape
    return torch.cat([w, torch.zeros(w.shape[0], CRITIC_NEW_EXTRA, dtype=w.dtype)], dim=1)


WIDEN = {"actor.0.weight": widen_actor, "critic.0.weight": widen_critic}
for k, f in WIDEN.items():
    sd[k] = f(sd[k])

# Keep the trained optimizer state (Adam moments + the adaptive learning rate!) and widen it the same way.
# A fresh Adam with lr=1e-3 wrecked the warm-started policy within ~30 iterations (load_v1, 2026-10-16).
opt = ck["optimizer_state_dict"]
assert len(opt["param_groups"][0]["params"]) == len(keys), (len(opt["param_groups"][0]["params"]), len(keys))
for i, k in enumerate(keys):
    if k in WIDEN and i in opt["state"]:
        for mom in ("exp_avg", "exp_avg_sq"):
            opt["state"][i][mom] = WIDEN[k](opt["state"][i][mom])
ck["optimizer_state_dict"] = opt
ck["model_state_dict"] = sd
torch.save(ck, dst)
print("saved", dst, {k: tuple(v.shape) for k, v in sd.items() if k.endswith("0.weight")},
      "lr", opt["param_groups"][0]["lr"])
