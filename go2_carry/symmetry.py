"""Left-right mirror symmetry for the Go2 biped tasks (RSL-RL data augmentation).

Follows Isaac Lab's ``locomotion/velocity/mdp/symmetry/anymal.py`` and Mittal et al. 2024 ("Symmetry
considerations for learning task symmetric robot policies"). Policies warm-started from the v1 chain kept a
0.7 rad left/right hip-pitch offset and curved right on straight commands; mirroring every sample makes the
learned policy (approximately) left-right symmetric.

Mirror plane = the body x-z plane (y -> -y). This holds in any posture, also standing on the hind legs.
Per observation term, keyed by term name; slices are built from the observation manager, so adding or
reordering terms does not silently break the mirroring (an unknown term raises).
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

# sign pattern applied after (optional) joint swap; "joints" = swap L/R legs and negate hip ab/adduction
_VEC_SIGNS = {
    "base_lin_vel": [1, -1, 1],          # polar vector in body frame
    "base_ang_vel": [-1, 1, -1],         # axial vector
    "projected_gravity": [1, -1, 1],
    "velocity_commands": [1, -1, -1],    # (vx, vy, wz) in the biped heading frame
    "biped_lin_vel": [1, -1],            # (forward, left)
    "base_height": [1],
    "gait_phase": [-1, -1],              # swapping legs = phase + 0.5 -> (sin, cos) negate
    "heading_error": [-1],               # yaw angle
    "box_rel_pos": [1, -1, 1],           # heading frame (fwd, left, up)
    "box_rel_vel": [1, -1, 1],
    "box_mass": [1],
    "box_present": [1],
    "obj_rel_pos": [1, -1, 1],           # heading frame (fwd, left, up)
    "obj_rel_vel": [1, -1, 1],
    "obj_mass": [1],
    "obj_present": [1],
    "obj_dims": [1, 1, 1],               # sizes are mirror-invariant
    "gait_mode": [1, 1],                 # [quad, biped] one-hot: mirror-invariant
}
_JOINT_TERMS = {"joint_pos_rel", "joint_vel_rel", "joint_effort", "last_action"}
# per-joint physical parameters (RefWalk critic DR terms): swap L/R legs, no sign flip (a gain is not an angle)
_JOINT_PARAM_TERMS = {"actuator_kp_scale", "actuator_kd_scale", "joint_friction", "joint_armature"}
_SWAP_TERMS = {"forearm_load": [1, 0]}  # (left, right) payload forces -> swap

_cache: dict = {}


def _joint_maps(env):
    if "perm" not in _cache:
        names = env.scene["robot"].joint_names
        perm, sign = [], []
        for n in names:
            # first char is F/R (front/rear), second L/R (side): FL_hip_joint <-> FR_hip_joint
            m = n[0] + {"L": "R", "R": "L"}[n[1]] + n[2:]
            perm.append(names.index(m))
            sign.append(-1.0 if "_hip_" in n else 1.0)
        _cache["perm"] = torch.tensor(perm, device=env.device)
        _cache["sign"] = torch.tensor(sign, device=env.device)
    return _cache["perm"], _cache["sign"]


def _mirror_group(env, group: str, x: torch.Tensor) -> torch.Tensor:
    key = ("slices", group)
    if key not in _cache:
        om = env.observation_manager
        names = om.active_terms[group]
        dims = om.group_obs_term_dim[group]
        slices, i = [], 0
        for n, d in zip(names, dims):
            k = int(torch.tensor(d).prod().item())
            if n in _VEC_SIGNS:
                base = len(_VEC_SIGNS[n])
            elif n in _SWAP_TERMS:
                base = len(_SWAP_TERMS[n])
            elif n in _JOINT_TERMS or n in _JOINT_PARAM_TERMS:
                base = 12
            else:
                raise KeyError(f"symmetry: no mirror rule for observation term '{n}' in group '{group}'")
            assert k % base == 0, (n, k, base)
            slices.append((n, i, i + k, base))  # k = history_length * base (flattened oldest -> newest)
            i += k
        _cache[key] = slices
    perm, sign = _joint_maps(env)
    y = x.clone()
    for n, a, b, base in _cache[key]:
        t = x[:, a:b].reshape(x.shape[0], -1, base)
        if n in _JOINT_TERMS:
            t = t[:, :, perm] * sign
        elif n in _JOINT_PARAM_TERMS:
            t = t[:, :, perm]
        elif n in _SWAP_TERMS:
            t = t[:, :, _SWAP_TERMS[n]]
        else:
            t = t * torch.tensor(_VEC_SIGNS[n], device=x.device, dtype=x.dtype)
        y[:, a:b] = t.reshape(x.shape[0], -1)
    return y


@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Returns (obs, actions) with the left-right mirrored copy appended (batch size x2)."""
    u = env.unwrapped
    obs_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for g in obs.keys():
            obs_aug[g][n:] = _mirror_group(u, g, obs[g])
    actions_aug = None
    if actions is not None:
        perm, sign = _joint_maps(u)
        actions_aug = torch.cat([actions, actions[:, perm] * sign], dim=0)
    return obs_aug, actions_aug
