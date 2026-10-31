"""General warm start: copy a trained policy into a task whose observations have extra terms / history.

Every observation group is described as a list of term dims. New terms are appended after the old ones;
with history H each term is flattened oldest -> newest (Isaac Lab), and old weights go into the newest
frame. All new input columns (and their Adam moments) start at zero, so the warm-started network behaves
exactly like the source one. The trained optimizer state, incl. the adaptive learning rate, is kept.

  python scripts/warmstart.py SRC DST --actor 3,3,3,12,12,12 --actor_new 2 --actor_hist 1 \
                                       --critic_old 63 --critic_new 2
"""
import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--actor", default="3,3,3,12,12,12", help="old actor term dims (no history in the source)")
p.add_argument("--actor_new", default="", help="dims of actor terms appended by the new task")
p.add_argument("--actor_hist", type=int, default=1)
p.add_argument("--critic_old", type=int, default=63)
p.add_argument("--critic_new", type=int, default=0)
a = p.parse_args()
old_terms = [int(x) for x in a.actor.split(",") if x]
new_terms = [int(x) for x in a.actor_new.split(",") if x]
H = a.actor_hist

ck = torch.load(a.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
keys = list(sd.keys())


def widen_actor(w):
    assert w.shape[1] == sum(old_terms), (w.shape, old_terms)
    out = torch.zeros(w.shape[0], sum(old_terms + new_terms) * H, dtype=w.dtype)
    o_old, o_new = 0, 0
    for i, d in enumerate(old_terms + new_terms):
        if i < len(old_terms):
            newest = o_new + (H - 1) * d
            out[:, newest:newest + d] = w[:, o_old:o_old + d]
            o_old += d
        o_new += H * d
    return out


def widen_critic(w):
    assert w.shape[1] == a.critic_old, w.shape
    return torch.cat([w, torch.zeros(w.shape[0], a.critic_new, dtype=w.dtype)], dim=1)


WIDEN = {"actor.0.weight": widen_actor, "critic.0.weight": widen_critic}
for k, f in WIDEN.items():
    sd[k] = f(sd[k])
opt = ck["optimizer_state_dict"]
assert len(opt["param_groups"][0]["params"]) == len(keys)
for i, k in enumerate(keys):
    if k in WIDEN and i in opt["state"]:
        for mom in ("exp_avg", "exp_avg_sq"):
            opt["state"][i][mom] = WIDEN[k](opt["state"][i][mom])
torch.save(ck, a.dst)
print("saved", a.dst, {k: tuple(v.shape) for k, v in sd.items() if k.endswith("0.weight")}, "lr", opt["param_groups"][0]["lr"])
