"""MDP terms for biped <-> quadruped switching (2026-10-31) and for the measured real-robot joints.

Switching follows go2_quad2hand (ref): one policy, a gait-mode one-hot in the observation, the mode re-drawn during the
episode (ref _resample_commands every 10 s), every mode-specific reward multiplied by (mode == its mode). No reward is
written for the transition itself; it comes from the target mode's posture terms. Here:
  * mode 0 = quadruped (stand still on four legs: a rest for the hot hind-leg motors), mode 1 = biped (everything trained
    so far). The one-hot is [quad, biped].
  * only without an object (user, 2026-10-31: go down only when the hands are empty): episodes with an object stay
    biped until it is taken away (box_mdp.Box.present False); from then on, or in episodes without an object, the mode
    flips every U(switch_s) seconds. Quadruped resets without an object start in the quadruped mode with p_quad_start.
  * the quadruped mode commands zero velocity; back in the biped mode the command term draws a new command.

Real joints (deploy/real_joints.py, measured on a real Go2): gearbox friction growing with the transmitted torque,
Fs + mu |tau|, and viscous Fd per joint type; none of it was in training before (the Go2HV actuator's Fs / Fd are 0). In
MuJoCo it slowed the Tripod 0.20 -> 0.06-0.09 m/s and tipped its walk gait over 3 of 6 times; HoldV4 c16 fell after
standing up (measured knee rotor inertia outside the trained 0.01-0.05).
"""
from __future__ import annotations

import functools
import os
import types

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

from . import box_mdp as hmdp
from . import mdp as bmdp

QUAD, BIPED = 0, 1


class Mode:
    quad: torch.Tensor | None = None       # (N,) bool
    t_next: torch.Tensor | None = None     # (N,) s: next flip
    had_obj: torch.Tensor | None = None    # (N,) bool: an object was present last step
    flips: int = 0


def _init(env):
    if Mode.quad is None or Mode.quad.shape[0] != env.num_envs:
        n, dev = env.num_envs, env.device
        Mode.quad = torch.zeros(n, dtype=torch.bool, device=dev)
        Mode.t_next = torch.full((n,), float("inf"), device=dev)
        Mode.had_obj = torch.zeros(n, dtype=torch.bool, device=dev)


def mask(env, mode: int) -> torch.Tensor:
    _init(env)
    return (Mode.quad if mode == QUAD else ~Mode.quad).float()


def gated(fn, mode: int):
    """A reward function that pays only in `mode` (ref: every mode-specific reward x (gait_mode == m)). functools.wraps
    keeps fn's signature, which Isaac Lab checks against the term's params."""
    if getattr(fn, "_gated_mode", None) is not None:
        return fn

    @functools.wraps(fn)
    def g(env, *args, **kwargs):
        return fn(env, *args, **kwargs) * mask(env, mode)

    g._gated_mode = mode
    return g


def biped_only(fn):
    """A termination that ends the episode only in the biped mode (bool & biped), signature kept (functools.wraps)."""
    @functools.wraps(fn)
    def g(env, *args, **kwargs):
        _init(env)
        return fn(env, *args, **kwargs) & ~Mode.quad

    return g


def mode_switch(env, env_ids, switch_s: tuple[float, float], p_quad_start: float, quad_up_max: float = 0.3,
                fixed_mode: int | None = None, fixed_switch_s: float | None = None) -> None:
    """Interval event (every step, after box_mdp.object_handover): draw / flip the mode and zero the quadruped command.
    fixed_mode (evaluation): every env starts in this mode; fixed_switch_s: object-free envs flip once at this time."""
    _init(env)
    dev = env.device
    t = env.episode_length_buf.float() * env.step_dt
    fresh = env.episode_length_buf <= 1
    obj = hmdp.Box.present | hmdp.Box.placed if hmdp.Box.present is not None else torch.zeros_like(Mode.quad)
    if fresh.any():
        k = int(fresh.sum())
        quad_pose = bmdp.uprightness(env)[fresh] < quad_up_max
        start = (~obj[fresh]) & quad_pose & (torch.rand(k, device=dev) < p_quad_start)
        if fixed_mode is not None:
            start = torch.full((k,), fixed_mode == QUAD, device=dev)
        Mode.quad[fresh] = start
        nxt = sample_uniform(*switch_s, (k,), dev) if fixed_switch_s is None else torch.full((k,), fixed_switch_s, device=dev)
        Mode.t_next[fresh] = t[fresh] + nxt
    # an object episode stays biped; the clock to the first flip starts when the object is taken away
    Mode.quad[obj] = False
    freed = Mode.had_obj & ~obj
    if freed.any():
        Mode.t_next[freed] = t[freed] + (sample_uniform(*switch_s, (int(freed.sum()),), dev) if fixed_switch_s is None
                                         else fixed_switch_s)
    Mode.had_obj = obj.clone()
    due = ~obj & ~fresh & (t >= Mode.t_next)
    if due.any():
        Mode.quad[due] = ~Mode.quad[due]
        Mode.flips += int(due.sum())
        Mode.t_next[due] = (t[due] + sample_uniform(*switch_s, (int(due.sum()),), dev) if fixed_switch_s is None
                            else torch.full_like(t[due], float("inf")))
        to_biped = (due & ~Mode.quad).nonzero(as_tuple=True)[0]
        if len(to_biped):
            env.command_manager.get_term("base_velocity")._resample_command(to_biped)
    cmd = env.command_manager.get_term("base_velocity")
    cmd.vel_command_b[Mode.quad] = 0.0


def gait_mode_onehot(env) -> torch.Tensor:
    """[quad, biped] (ref gait_mode one-hot: 0 quadruped, 1 hand)."""
    _init(env)
    return torch.stack([Mode.quad.float(), (~Mode.quad).float()], dim=1)


def feet_contact_frac(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Fraction of the listed feet in contact (quadruped stance: all four down)."""
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    return (f > threshold).float().mean(dim=1)


def body_level(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """cos of the body tilt from level, clamped at 0: 0 upright (pitch 90 deg) -> 1 on four legs, linear in cos."""
    return (-env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]).clamp(min=0.0)


# ----------------------------------------------------------------------------- real joints
def _real_joints() -> dict:
    """Fs, mu, Fd per joint type from deploy/real_joints.py (measured values in real_joints_local.json, not in git)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "real_joints.py")
    spec = importlib.util.spec_from_file_location("go2_real_joints", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: v[:3] for k, v in mod.REAL_JOINTS.items()}


REAL_JOINTS = _real_joints()  # Fs, mu, Fd


def _compute_with_load_friction(self, control_action, joint_pos, joint_vel):
    """UnitreeActuator.compute + gearbox friction growing with the clipped motor torque (mu |tau| tanh(v / Va))."""
    out = type(self).compute(self, control_action, joint_pos, joint_vel)
    tau = self.applied_effort
    mu = getattr(self, "_friction_load", None)
    if mu is not None:
        self.applied_effort = tau - mu * tau.abs() * torch.tanh(joint_vel / self._activation_vel)
        out.joint_efforts = self.applied_effort
    return out


def randomize_joint_friction_real(env, env_ids, scale_range: tuple[float, float],
                                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Reset event: per env and joint type, Fs / mu / Fd = measured x U(scale_range), on every Unitree actuator."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for act in asset.actuators.values():
        if not hasattr(act, "_friction_static"):
            continue
        if not hasattr(act, "_friction_load"):
            act._friction_load = torch.zeros_like(act._friction_static)
            act.compute = types.MethodType(_compute_with_load_friction, act)
            names = act.joint_names
            act._fric_base = torch.tensor([REAL_JOINTS[n.split("_")[1]] for n in names], device=env.device)  # (J, 3)
            print(f"[switch] measured joint friction x{scale_range} on {names}", flush=True)
        s = torch.empty(len(env_ids), act._fric_base.shape[0], 3, device=env.device).uniform_(*scale_range)
        v = act._fric_base[None] * s
        act._friction_static[env_ids] = v[..., 0]
        act._friction_load[env_ids] = v[..., 1]
        act._friction_dynamic[env_ids] = v[..., 2]
