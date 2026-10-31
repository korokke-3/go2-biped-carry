"""Stage 1 of real-object carrying: receive an object on the front legs while standing, and keep it.

- Objects (one per env, env i gets shape i % K, so every env knows its object): large boxes that span both
  forearms ("tray" support) down to small ones narrower than the 0.28 m arm gap that need the arms folded around
  them. Contact compliance per shape: rigid / cardboard-like / soft (PhysX compliant contacts).
- Hand-over: an object episode gets a zero velocity command; the object is placed once the front legs are close to
  the tray pose, or after ``max_wait_s`` anyway (no "refuse to take it" loophole). Gentle or dropped from <= 10 cm.
- Curriculum (adaptive, like the command curriculum): the max object mass grows 0.3 -> 2.0 kg while the drop rate
  is low; at 2 kg the next shape level (smaller objects) is unlocked and the mass restarts at 1 kg.
- Dropping the object ends the episode.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from . import mdp as bmdp

# (fwd depth, lateral width, height) [m], contact stiffness [N/m] (0 = rigid), damping [N s/m], curriculum level
BOX_SHAPES = [
    ((0.20, 0.36, 0.12), 0.0, 0.0, 0),      # large rigid box
    ((0.25, 0.40, 0.16), 5000.0, 100.0, 0),  # large cardboard-like
    ((0.15, 0.32, 0.08), 1000.0, 60.0, 0),   # flat soft
    ((0.18, 0.30, 0.14), 0.0, 0.0, 1),      # just wider than the arm gap
    ((0.16, 0.24, 0.12), 5000.0, 100.0, 2),  # narrower than the gap: needs folded / closed arms
    ((0.14, 0.18, 0.12), 0.0, 0.0, 3),      # small rigid
    ((0.12, 0.14, 0.10), 1000.0, 60.0, 99),  # small soft: disabled (99) - dropped 75-100 % in every carry_walk_v3
    #                                          test (half the 0.28 m arm gap); kept in the evaluation, not trained
    ((0.16, 0.16, 0.16), 5000.0, 100.0, 3),  # cube
]
DISABLED_LEVEL = 99
MAX_LEVEL = max(s[3] for s in BOX_SHAPES if s[3] < DISABLED_LEVEL)
FRONT_JOINTS = ["FL_hip_joint", "FR_hip_joint", "FL_thigh_joint", "FR_thigh_joint", "FL_calf_joint", "FR_calf_joint"]
# calf -1.45: forearm tips ~9 deg up, so an object slides toward the chest instead of off the tips (diag_hold_placement)
# hold_v7 -> v8: calf -1.45 -> -1.6 (forearm tips ~17 deg up). The large deep box (0.25 m, overhanging the forearm
# tips by 5 cm) rested on one forearm and slid FORWARD off the tips within ~1 s (diag_hold_handover, 2 kg).
TRAY_TARGET = [0.0, 0.0, 1.3, 1.3, -1.6, -1.6]
BELLY_FWD = 0.057  # trunk half-thickness: the belly surface in front of the base when upright


class Box:
    present: torch.Tensor | None = None   # (N,) bool: this episode gets an object
    tray_walk: torch.Tensor | None = None  # (N,) bool: no object, but walk with the arms held in the tray pose
    placed: torch.Tensor | None = None    # (N,) bool
    t_ready: torch.Tensor | None = None   # (N,) s: hand-over possible from here
    drop: torch.Tensor | None = None      # (N,) m extra drop height
    dims: torch.Tensor | None = None      # (N, 3)
    level_of: torch.Tensor | None = None  # (N,) curriculum level of the env's shape
    level: int = 0
    m_max: float = 0.3
    cmd_scale: float = 0.0  # stage 2: fraction of the velocity command given to object episodes
    cmd_raw: torch.Tensor | None = None
    cmd_written: torch.Tensor | None = None
    walk: bool = False      # stage 2: once the object curriculum is complete, grow cmd_scale by the drop rate
    assist: float = 1.0     # stabilising-force curriculum (1 = full, 0 = none); mass / shape levels wait for 0
    handovers: int = 0
    drops: int = 0
    # HoldRef (2026-10-20): slow hand-over / removal by a "hand" = vertical support force on the object
    t_place: torch.Tensor | None = None    # (N,) s: time the object was placed
    release_T: torch.Tensor | None = None  # (N,) s: support force m g -> 0 over this time (0 = put down at once)
    t_remove: torch.Tensor | None = None   # (N,) s: removal starts (inf = never)
    remove_T: torch.Tensor | None = None   # (N,) s: lifting force 0 -> m g over this time, then the object is gone
    removals: int = 0
    # HoldV4 (2026-10-22) overrides; defaults = Hold / HoldRef behaviour
    tray_target: list | None = None        # waiting pose of the front legs (FRONT_JOINTS order); None -> TRAY_TARGET
    max_level: int | None = None           # last curriculum level; None -> MAX_LEVEL
    m_cap: float = 2.0                     # mass limit; above 2 kg the mass grows only after walking (cmd_scale 1)
    size_table: dict | None = None         # {"dims": (N,3) cpu, "level": (N,) cpu, "ref": (N,) cpu or None}
    ref_id: torch.Tensor | None = None     # (N,) evaluation reference object id (fixed-size evaluation set)


def _init(env):
    n, dev = env.num_envs, env.device
    if Box.present is None or Box.present.shape[0] != n:
        k = len(BOX_SHAPES)
        idx = torch.arange(n, device=dev) % k
        Box.present = torch.zeros(n, dtype=torch.bool, device=dev)
        Box.tray_walk = torch.zeros(n, dtype=torch.bool, device=dev)
        Box.placed = torch.zeros(n, dtype=torch.bool, device=dev)
        Box.t_ready = torch.zeros(n, device=dev)
        Box.drop = torch.zeros(n, device=dev)
        Box.dims = torch.tensor([s[0] for s in BOX_SHAPES], device=dev)[idx]
        Box.level_of = torch.tensor([s[3] for s in BOX_SHAPES], device=dev)[idx]
        if Box.size_table is not None:  # HoldV4: per-env random sizes (randomize_object_sizes)
            Box.dims = Box.size_table["dims"].to(dev)
            Box.level_of = Box.size_table["level"].to(dev)
            Box.ref_id = None if Box.size_table.get("ref") is None else Box.size_table["ref"].to(dev)
        Box.t_place = torch.zeros(n, device=dev)
        Box.release_T = torch.zeros(n, device=dev)
        Box.t_remove = torch.full((n,), float("inf"), device=dev)
        Box.remove_T = torch.zeros(n, device=dev)


def active(env) -> torch.Tensor:
    _init(env)
    return Box.placed


def _front_dev(env) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    ids = robot.find_joints(FRONT_JOINTS, preserve_order=True)[0]
    return (robot.data.joint_pos[:, ids] - torch.tensor(_tray(), device=env.device)).abs().sum(dim=1)


def _tray():
    return TRAY_TARGET if Box.tray_target is None else Box.tray_target


def _max_level():
    return MAX_LEVEL if Box.max_level is None else Box.max_level


# ----------------------------------------------------------------------------- events
def object_mass_curriculum(env, env_ids: torch.Tensor, fixed_mass: float | None = None, p_base: float = 0.0) -> None:
    """Reset event: object mass ~ U(0.1, Box.m_max) (or fixed for evaluation); inertia scaled with the mass."""
    _init(env)
    obj = env.scene["object"]
    ids = env_ids.cpu()
    masses = obj.root_physx_view.get_masses()
    m = (torch.full((len(ids),), fixed_mass) if fixed_mass is not None
         else 0.1 + torch.rand(len(ids)) * max(Box.m_max - 0.1, 0.0))
    if fixed_mass is None and p_base > 0.0 and Box.m_max > 2.0:  # keep practising the common masses (<= 2 kg)
        low = torch.rand(len(ids)) < p_base
        m[low] = 0.1 + torch.rand(int(low.sum())) * 1.9
    ratio = m / obj.data.default_mass[ids, 0].cpu()
    masses[ids, 0] = m
    obj.root_physx_view.set_masses(masses, ids)
    inertias = obj.root_physx_view.get_inertias()
    inertias[ids] = obj.data.default_inertia[ids].cpu() * ratio[:, None]
    obj.root_physx_view.set_inertias(inertias, ids)


def object_handover(env, env_ids: torch.Tensor, p_object: float, p_tray_walk: float, ready_range_s: tuple[float, float],
                    max_wait_s: float, ready_tol: float, p_gentle: float, drop_max: float,
                    park_height: float = 3.0, fixed_level: int | None = None, fixed_drop: float | None = None,
                    fixed_ready_s: float | None = None, obj_cmd_scale: float | None = None,
                    release_range_s: tuple[float, float] = (0.0, 0.0), place_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    p_remove: float = 0.0, remove_after_s: tuple[float, float] = (3.0, 10.0),
                    remove_ramp_s: tuple[float, float] = (0.0, 1.0), ready_gate_while_assist: bool = False,
                    fixed_release_s: float | None = None, fixed_remove_s: float | None = None) -> None:
    """Interval event (every step): hand-over logic, zero command for object episodes, parking.

    HoldRef additions (defaults = the old behaviour):
    * ``release_range_s``: gentle hand-overs (drop 0) are released slowly - a vertical support force goes from m g to 0
      over T ~ U(range) (task: "the object may be handed over slowly or put down suddenly"). Dropped ones: T = 0.
    * ``place_noise`` (fwd [0, a], lateral +-b, yaw +-c): a person does not place the object exactly centred.
    * ``p_remove``: the object is taken away remove_after_s after placement, lifted with a force 0 -> m g over
      remove_ramp_s (0 = snatched); afterwards the episode continues without object (arms back to the tray pose).
    * ``ready_gate_while_assist``: wait for the arms (ready_tol) only while the assist curriculum is still > 0; later
      the object comes at t_ready regardless (the person decides, the policy cannot delay it)."""
    _init(env)
    obj = env.scene["object"]
    n, dev = env.num_envs, env.device
    level = Box.level if fixed_level is None else fixed_level
    fresh = env.episode_length_buf <= 1
    k = int(fresh.sum())
    if k:
        Box.present[fresh] = (torch.rand(k, device=dev) < p_object) & (Box.level_of[fresh] <= level)
        # the arms weigh ~1 kg each: holding the tray pose is a skill of its own, also while walking without an object
        Box.tray_walk[fresh] = ~Box.present[fresh] & (torch.rand(k, device=dev) < p_tray_walk)
        Box.placed[fresh] = False
        Box.t_remove[fresh] = float("inf")
        Box.t_ready[fresh] = (sample_uniform(*ready_range_s, (k,), dev) if fixed_ready_s is None
                              else torch.full((k,), fixed_ready_s, device=dev))
        d = sample_uniform(0.0, drop_max, (k,), dev)
        d = torch.where(torch.rand(k, device=dev) < p_gentle, torch.zeros_like(d), d)
        Box.drop[fresh] = d if fixed_drop is None else torch.full((k,), fixed_drop, device=dev)
        # reference-state initialisation (cf. DeepMimic, Peng et al. 2018): object episodes that start upright start
        # with the front legs already in the tray pose, so holding is experienced from the start (hold_v1/v2 never
        # found the tray pose on their own: every hand-over ended in a drop)
        rsi = fresh & Box.present & (bmdp.uprightness(env) > 0.8)
        if rsi.any():
            robot: Articulation = env.scene["robot"]
            ids = rsi.nonzero(as_tuple=True)[0]
            jid = robot.find_joints(FRONT_JOINTS, preserve_order=True)[0]
            q = robot.data.joint_pos[ids].clone()
            qd = robot.data.joint_vel[ids].clone()
            q[:, jid] = torch.tensor(_tray(), device=dev) + sample_uniform(-0.1, 0.1, (len(ids), len(jid)), dev)
            qd[:, jid] = 0.0
            robot.write_joint_state_to_sim(q, qd, env_ids=ids)
    # object episodes: velocity command scaled by Box.cmd_scale (stage 1: 0 = stand and receive; stage 2 grows it)
    # the command term keeps its own (resampled) command; remember the raw one and write raw * scale, so the scale is
    # applied once and not compounded every step
    cmd = env.command_manager.get_term("base_velocity")
    scale = Box.cmd_scale if obj_cmd_scale is None else obj_cmd_scale
    if Box.cmd_raw is None or Box.cmd_raw.shape[0] != n:
        Box.cmd_raw = cmd.vel_command_b.clone()
        Box.cmd_written = cmd.vel_command_b.clone()
    resampled = (cmd.vel_command_b != Box.cmd_written).any(dim=1)
    Box.cmd_raw[resampled] = cmd.vel_command_b[resampled]
    Box.cmd_raw[~Box.present] = cmd.vel_command_b[~Box.present]
    out = cmd.vel_command_b.clone()
    out[Box.present] = Box.cmd_raw[Box.present] * scale
    out[Box.present & ~Box.placed] = 0.0  # wait for the object standing still
    cmd.vel_command_b[:] = out
    Box.cmd_written = out.clone()
    t = env.episode_length_buf.float() * env.step_dt
    upright = bmdp.uprightness(env) > 0.8
    gate = (not ready_gate_while_assist) or Box.assist > 0
    ready = (_front_dev(env) < ready_tol) | (t >= Box.t_ready + max_wait_s) if gate else torch.ones_like(Box.placed)
    due = Box.present & ~Box.placed & (t >= Box.t_ready) & upright & ready
    parked = ~Box.placed & ~due
    if parked.any():
        ids = parked.nonzero(as_tuple=True)[0]
        pose = torch.zeros(len(ids), 7, device=dev)
        pose[:, :3] = env.scene.env_origins[ids] + torch.tensor([0.0, 0.0, park_height], device=dev)
        pose[:, 3] = 1.0
        obj.write_root_pose_to_sim(pose, env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)
    if due.any():
        ids = due.nonzero(as_tuple=True)[0]
        robot: Articulation = env.scene["robot"]
        fwd, left = bmdp._heading_frame(env)
        c, top = bmdp.forearm_center_w(env)
        dims = Box.dims[ids]
        lat = ((c - robot.data.root_pos_w) * left).sum(1)[ids]
        m_ = len(ids)
        place_fwd = BELLY_FWD + dims[:, 0] / 2 + 0.01 + sample_uniform(0.0, place_noise[0], (m_,), dev)  # back face >= 1 cm off the belly
        lat = lat + sample_uniform(-place_noise[1], place_noise[1], (m_,), dev)
        pos = robot.data.root_pos_w[ids] + place_fwd[:, None] * fwd[ids] + lat[:, None] * left[ids]
        pos[:, 2] = top[ids] + dims[:, 2] / 2 + Box.drop[ids]
        yaw = bmdp._biped_yaw(env)[ids] + sample_uniform(-place_noise[2], place_noise[2], (m_,), dev)
        z = torch.zeros_like(yaw)
        obj.write_root_pose_to_sim(torch.cat([pos, quat_from_euler_xyz(z, z, yaw)], 1), env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)
        Box.placed[ids] = True
        Box.handovers += len(ids)
        Box.t_place[ids] = t[ids]
        rT = (sample_uniform(*release_range_s, (m_,), dev) if fixed_release_s is None
              else torch.full((m_,), fixed_release_s, device=dev))
        Box.release_T[ids] = torch.where(Box.drop[ids] > 0, torch.zeros_like(rT), rT)
        rem = torch.rand(m_, device=dev) < p_remove
        if fixed_remove_s is not None:
            rem[:] = True
        after = (sample_uniform(*remove_after_s, (m_,), dev) if fixed_remove_s is None
                 else torch.full((m_,), fixed_remove_s, device=dev))
        Box.t_remove[ids] = torch.where(rem, t[ids] + after, torch.full_like(after, float("inf")))
        Box.remove_T[ids] = sample_uniform(*remove_ramp_s, (m_,), dev)
    # removal finished -> the object is gone (parked next step), the episode goes on without it
    gone = Box.placed & (t >= Box.t_remove + Box.remove_T)
    if gone.any():
        Box.placed[gone] = False
        Box.present[gone] = False
        Box.tray_walk[gone] = True
        Box.t_remove[gone] = float("inf")
        Box.removals += int(gone.sum())
        ids = gone.nonzero(as_tuple=True)[0]
        pose = torch.zeros(len(ids), 7, device=dev)
        pose[:, :3] = env.scene.env_origins[ids] + torch.tensor([0.0, 0.0, park_height], device=dev)
        pose[:, 3] = 1.0
        obj.write_root_pose_to_sim(pose, env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)
    _assist_forces(env, obj, support=_hand_support(env, t))


def _hand_support(env, t: torch.Tensor) -> torch.Tensor:
    """Fraction of the object weight carried by the person's hand: 1 -> 0 over release_T after placement,
    0 -> 1 over remove_T from t_remove. (N,)"""
    rel = torch.where(Box.release_T > 0, 1.0 - (t - Box.t_place) / Box.release_T.clamp(min=1e-6), torch.zeros_like(t))
    lift = torch.where(Box.remove_T > 0, (t - Box.t_remove) / Box.remove_T.clamp(min=1e-6), torch.zeros_like(t))
    lift = torch.where(torch.isfinite(Box.t_remove), lift, torch.zeros_like(t))
    return torch.maximum(rel, lift).clamp(0.0, 1.0) * Box.placed.float()


def _assist_forces(env, obj, omega: float = 10.0, ang_rate: float = 10.0, support: torch.Tensor | None = None) -> None:
    """Assistive-force curriculum (cf. Yu, Turk & Liu 2018): while the object is held, a horizontal spring-damper
    pulls it over the forearm centre and a damper resists its rotation, scaled by Box.assist and the object mass.
    NO vertical support: the arms still have to be under the object (an earlier weightless assist taught the carry
    policy not to touch the object). hold_v3: the object rested on one forearm only and rolled off in 0.2-0.5 s.
    The wrench is computed once per policy step (0.02 s) and held over the substeps, so the gains are kept explicit-
    stable: hold_v4 used an angular damping of 0.5 * m, i.e. 2.3 x per step for the small cube -> it span up and the
    value loss exploded. Now omega 10 rad/s (d/m * dt = 0.4) and an angular rate of 10/s scaled by the inertia
    (0.2 per step)."""
    n, dev = env.num_envs, env.device
    force = torch.zeros(n, 1, 3, device=dev)
    torque = torch.zeros(n, 1, 3, device=dev)
    a = Box.assist
    if a > 0:
        on = Box.placed.float()[:, None]
        m = bmdp.object_mass(env)  # (N, 1)
        c, _ = bmdp.forearm_center_w(env)
        robot: Articulation = env.scene["robot"]
        k = a * m * omega**2
        d = a * m * 2.0 * omega
        dp = (c - obj.data.root_pos_w)[:, :2]
        dv = (robot.data.root_lin_vel_w - obj.data.root_lin_vel_w)[:, :2]
        force[:, 0, :2] = (k * dp + d * dv) * on
        inertia = obj.root_physx_view.get_inertias().to(dev).reshape(n, -1)[:, [0, 4, 8]]  # body-frame diagonal
        torque[:, 0, :] = -a * ang_rate * inertia.min(dim=1, keepdim=True)[0] * obj.data.root_ang_vel_w * on
    if support is not None:  # the person's hand (slow hand-over / removal): vertical, independent of the assist
        force[:, 0, 2] += support * bmdp.object_mass(env)[:, 0] * 9.81
    obj.set_external_force_and_torque(force, torque, is_global=True)


# ----------------------------------------------------------------------------- curriculum
def _status(env, rate):
    """Append the curriculum state to $HOLD_STATUS_FILE (the logged curriculum value is an average and hides it)."""
    import json
    import os
    f = os.environ.get("HOLD_STATUS_FILE")
    if f:
        with open(f, "a") as fh:
            fh.write(json.dumps({"step": int(env.common_step_counter), "rate": round(rate, 3), "level": Box.level,
                                 "m_max": round(Box.m_max, 2), "assist": Box.assist, "cmd_scale": Box.cmd_scale}) + "\n")


def object_curriculum(env, env_ids, min_handovers: int = 400, up_rate: float = 0.35, down_rate: float = 0.5,
                      assist_down: float = 0.4, assist_up: float = 0.6, start: dict | None = None):
    """Every episode length: first the assist is reduced (drop rate < 0.2: -0.1, > 0.5: +0.1); once it is 0, the drop
    rate decides harder (mass +0.2 kg; at 2 kg the next shape level, mass back to 1 kg) or easier (mass -0.2 kg).
    Returns level + mass / 10 + assist / 100 for logging.
    Thresholds (hold_v5 -> v6): training drop rates include up-to-10 cm drops, pushes and low friction; at assist 0 and
    a gentle hand-over model_19000 already dropped 0 % of the level-0 boxes while training still read ~20 %, so the
    assist step-down was 0.2 -> 0.3 and the mass step-up 0.1 -> 0.2. ``start`` (level / m_max / assist) restores the
    curriculum state when a run is resumed (Box is process state).
    hold_v6 -> v7: the training drop rate is measured with PPO's sampled actions. scripts/diagnostics/diag_hold_counter.py on
    model_20400 (training env, assist 0): 5-7 % with the mean action, 24-28 % with sampled actions, so all
    thresholds were raised by ~0.2 (assist 0.3/0.5 -> 0.4/0.6, mass 0.2/0.4 -> 0.35/0.5)."""
    if start is not None and not getattr(Box, "_started", False):
        Box.level, Box.m_max, Box.assist = start.get("level", Box.level), start.get("m_max", Box.m_max), start.get("assist", Box.assist)
        Box.cmd_scale, Box.walk = start.get("cmd_scale", Box.cmd_scale), start.get("walk", Box.walk)
        Box._started = True
    if env.common_step_counter % env.max_episode_length == 0 and Box.handovers >= min_handovers:
        rate = Box.drops / max(Box.handovers, 1)
        _status(env, rate)
        env.extras.setdefault("log", {})["Metrics/object_drop_rate"] = rate
        env.extras["log"]["Metrics/object_assist"] = Box.assist
        if Box.assist > 0 or rate > assist_up:
            if rate < assist_down:
                Box.assist = max(0.0, round(Box.assist - 0.1, 3))
            elif rate > assist_up:
                Box.assist = min(1.0, round(Box.assist + 0.1, 3))
            Box.handovers, Box.drops = 0, 0
            return torch.tensor(Box.level + Box.m_max / 10.0 + Box.assist / 100.0, device=env.device)
        if Box.walk and Box.level >= _max_level() and Box.m_max >= 2.0 - 1e-6:
            # stage 2: walking with the object, command share 0 -> 1 in steps of 0.1; then (HoldV4) mass beyond 2 kg
            if rate < up_rate and Box.cmd_scale >= 1.0 - 1e-6 and Box.m_max < Box.m_cap - 1e-6:
                Box.m_max = min(Box.m_cap, round(Box.m_max + 0.2, 3))
            elif rate < up_rate:
                Box.cmd_scale = min(1.0, round(Box.cmd_scale + 0.1, 3))
            elif rate > down_rate and Box.m_max > 2.0 + 1e-6:  # heavier than 2 kg fails: back off the mass first
                Box.m_max = max(2.0, round(Box.m_max - 0.2, 3))
            elif rate > down_rate:
                Box.cmd_scale = max(0.0, round(Box.cmd_scale - 0.1, 3))
            Box.handovers, Box.drops = 0, 0
            return torch.tensor(Box.level + Box.m_max / 10.0 + Box.cmd_scale / 1000.0 + 10.0, device=env.device)
        if rate < up_rate:
            if Box.m_max < 2.0 - 1e-6:
                Box.m_max = min(2.0, Box.m_max + 0.2)
            elif Box.level < _max_level():
                Box.level += 1
                Box.m_max = 1.0
        elif rate > down_rate:
            Box.m_max = max(0.3, Box.m_max - 0.2)
        env.extras.setdefault("log", {})["Metrics/object_drop_rate"] = rate
        Box.handovers, Box.drops = 0, 0
    return torch.tensor(Box.level + Box.m_max / 10.0 + Box.assist / 100.0, device=env.device)


# ----------------------------------------------------------------------------- observations (critic)
def _rel(env, v_w):
    fwd, left = bmdp._heading_frame(env)
    return torch.stack([(v_w * fwd).sum(1), (v_w * left).sum(1), v_w[:, 2]], 1)


def obj_rel_pos(env) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return _rel(env, env.scene["object"].data.root_pos_w - robot.data.root_pos_w) * active(env)[:, None]


def obj_rel_vel(env) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return _rel(env, env.scene["object"].data.root_lin_vel_w - robot.data.root_lin_vel_w) * active(env)[:, None]


def obj_mass(env) -> torch.Tensor:
    return bmdp.object_mass(env) * active(env)[:, None]


def obj_present(env) -> torch.Tensor:
    return active(env).float()[:, None]


def obj_dims(env) -> torch.Tensor:
    _init(env)
    return Box.dims * active(env)[:, None]


# ----------------------------------------------------------------------------- rewards / terminations
def holding(env) -> torch.Tensor:
    """1 per step while the object is held (dropping ends the episode, so this is the 'keep holding' value)."""
    return active(env).float() * bmdp._upright_gate(env)


def obj_shake(env) -> torch.Tensor:
    return torch.sum(torch.square(obj_rel_vel(env)), dim=1).clamp(max=4.0)  # capped: a flying object must not dominate


def obj_tilt(env) -> torch.Tensor:
    obj = env.scene["object"]
    up = quat_apply(obj.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    tilt = (1.0 - up[:, 2]) * active(env)
    if Box.size_table is not None and "is_sphere" in Box.size_table:  # a sphere rolls: its orientation is meaningless
        tilt = tilt * (~Box.size_table["is_sphere"].to(env.device)).float()
    return tilt


def fold_arms(env) -> torch.Tensor:
    """With an object: bend the forearms in from the tray pose (calf -1.3 -> -2.3), as far as the object allows."""
    robot: Articulation = env.scene["robot"]
    ids = robot.find_joints(["FL_calf_joint", "FR_calf_joint"], preserve_order=True)[0]
    f = ((-robot.data.joint_pos[:, ids] - 1.3) / 1.0).clamp(0.0, 1.0).mean(dim=1)
    return f * active(env)


def ready_pose(env) -> torch.Tensor:
    """L1 distance of the front legs from the tray pose: while waiting for the hand-over, and in tray-walk episodes."""
    _init(env)
    on = (Box.present & ~Box.placed) | Box.tray_walk
    return _front_dev(env) * on * bmdp._upright_gate(env)


def obj_dropped(env, min_height: float, max_dist: float) -> torch.Tensor:
    obj = env.scene["object"]
    robot: Articulation = env.scene["robot"]
    low = (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) < min_height
    far = (obj.data.root_pos_w - robot.data.root_pos_w)[:, :2].norm(dim=1) > max_dist
    out = (low | far) & active(env)
    Box.drops += int(out.sum())
    return out


def com_over_feet_with_obj(env, std: float) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    m = robot.root_physx_view.get_masses().to(env.device)
    xy = (m[..., None] * robot.data.body_com_pos_w[..., :2]).sum(dim=1)
    tot = m.sum(dim=1, keepdim=True)
    bm = bmdp.object_mass(env) * active(env).float()[:, None]
    xy = (xy + bm * env.scene["object"].data.root_pos_w[:, :2]) / (tot + bm)
    feet = robot.data.body_pos_w[:, robot.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)[0], :2].mean(dim=1)
    return torch.exp(-torch.sum(torch.square(xy - feet), dim=1) / std**2) * bmdp._upright_gate(env)


def obj_squeeze(env, sensor_name: str, factor: float) -> torch.Tensor:
    """Squeeze guard (real soft objects can break): sum of contact force magnitudes between the object and the
    front legs (per filtered body, so opposite pushes do not cancel), above ``factor`` x the object weight, squared
    relative to the weight (linear, capped at 3)."""
    f = env.scene.sensors[sensor_name].data.force_matrix_w  # (N, 1, n_filters, 3)
    total = f.norm(dim=-1).reshape(env.num_envs, -1).sum(dim=1)
    w = bmdp.object_mass(env)[:, 0] * 9.81
    # linear and capped: the contact impulse at hand-over / drop spiked the squared version to -105 per episode (hold_v1)
    return ((total - factor * w) / w.clamp(min=1.0)).clamp(0.0, 3.0) * active(env)


def knee_on_ground(env, sensor_names: list[str], threshold: float = 1.0) -> torch.Tensor:
    """Termination: a hind thigh / calf touches the GROUND while upright. hold_v8 knelt 48 % of the time when holding
    (2 feet + knee = a wider support), carry_walk_v3 35 %; v6b brushed the ground briefly while walking. A penalty
    of -1 (undesired_contacts) did not stop it, so it ends the episode like a fall. Not while standing up from four
    legs (uprightness <= 0.8), where the knees may touch."""
    return bmdp.body_touches_ground(env, sensor_names, threshold) & (bmdp.uprightness(env) > 0.8)


def knee_down_too_long(env, sensor_names: list[str], max_s: float, threshold: float = 1.0) -> torch.Tensor:
    """Termination (HoldRef, off by default): a hind thigh / calf has been on the GROUND for more than ``max_s``
    without a break while upright. Brief touches that the policy recovers from are allowed (user, 2026-10-20);
    kneeling as a third support (hold_v8 48 %, carry_walk_v3 35 % of the time) is not."""
    touch = bmdp.body_touches_ground(env, sensor_names, threshold) & (bmdp.uprightness(env) > 0.8)
    t = getattr(env, "_knee_down_s", None)
    if t is None or t.shape[0] != env.num_envs:
        t = torch.zeros(env.num_envs, device=env.device)
    t = torch.where(touch & (env.episode_length_buf > 0), t + env.step_dt, torch.zeros_like(t))
    env._knee_down_s = t
    return t > max_s


# ----------------------------------------------------------------------------- HoldV4: random object sizes
# unit-size base assets (asset index = env % V4_N_ASSETS): boxes 1 x 1 x 1 m first, then spheres of diameter 1 m (rigid / cardboard /
# soft each); the USD scale of each env's object sets its size before the simulation starts.
V4_N_ASSETS = 4  # 3 boxes (rigid / cardboard / soft) + 1 sphere (cardboard-like); was 3 + 3 until chunk 10
V4_N_BOX = 3
# evaluation reference set: the 8 old Hold boxes + spheres of 0.30 / 0.36 / 0.44 m
V4_REF_BOXES = [s_[0] for s_ in BOX_SHAPES]
V4_REF_SPHERES = [0.30, 0.36, 0.44]


def width_level(w: torch.Tensor, w_start: float, w_step: float, max_level: int) -> torch.Tensor:
    """Curriculum level that unlocks an object of lateral size w: level 0 = w >= w_start, +1 per w_step narrower."""
    return torch.ceil(((w_start - w) / w_step).clamp(min=0.0)).clamp(max=max_level).long()


def randomize_object_sizes(env, env_ids, box_fwd: tuple, box_lat: tuple, box_up: tuple, sphere_d: tuple,
                           w_start: float, w_step: float, max_level: int, fixed_refs: bool = False,
                           n_assets: int = V4_N_ASSETS, n_box: int = V4_N_BOX,
                           sphere_big: tuple | None = None, p_sphere_big: float = 0.0,
                           object_cfg=SceneEntityCfg("object")) -> None:
    """Prestartup (mode "usd", replicate_physics=False): per-env USD scale of the object = its size [m].
    Boxes: (fwd depth, lateral width, height) ~ U(ranges); spheres: diameter ~ U(sphere_d).
    fixed_refs: evaluation set instead (boxes cycle through V4_REF_BOXES, spheres through V4_REF_SPHERES).
    sphere_big / p_sphere_big: a share of the spheres is drawn from this (large) range instead of sphere_d.
    n_assets / n_box: base-asset layout of the scene (asset index = env % n_assets, the first n_box are boxes).
    Sets Box.size_table (dims, curriculum level from the lateral size, reference id) and Box.max_level."""
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, Sdf, UsdGeom, Vt

    import isaaclab.sim as sim_utils

    stage = get_current_stage()
    obj = env.scene[object_cfg.name]
    paths = sim_utils.find_matching_prim_paths(obj.cfg.prim_path)
    n = len(paths)
    idx = torch.arange(n)
    is_sphere = (idx % n_assets) >= n_box
    dims = torch.zeros(n, 3)
    ref = torch.full((n,), -1, dtype=torch.long)
    if fixed_refs:
        k = idx // n_assets
        rb = torch.tensor(V4_REF_BOXES)
        dims[~is_sphere] = rb[k[~is_sphere] % len(V4_REF_BOXES)]
        ref[~is_sphere] = k[~is_sphere] % len(V4_REF_BOXES)
        d = torch.tensor(V4_REF_SPHERES)[k[is_sphere] % len(V4_REF_SPHERES)]
        dims[is_sphere] = d[:, None].repeat(1, 3)
        ref[is_sphere] = len(V4_REF_BOXES) + k[is_sphere] % len(V4_REF_SPHERES)
    else:
        nb, ns = int((~is_sphere).sum()), int(is_sphere.sum())
        dims[~is_sphere] = torch.stack([torch.empty(nb).uniform_(*box_fwd), torch.empty(nb).uniform_(*box_lat),
                                        torch.empty(nb).uniform_(*box_up)], 1)
        d = torch.empty(ns).uniform_(*sphere_d)
        if sphere_big is not None:
            big = torch.rand(ns) < p_sphere_big
            d[big] = torch.empty(int(big.sum())).uniform_(*sphere_big)
        dims[is_sphere] = d[:, None].repeat(1, 3)
    with Sdf.ChangeBlock():
        for i in range(n):
            spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), paths[i])
            sc = spec.GetAttributeAtPath(paths[i] + ".xformOp:scale")
            new = sc is None
            if new:
                sc = Sdf.AttributeSpec(spec, paths[i] + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
            sc.default = Gf.Vec3f(*dims[i].tolist())
            if new:
                op = spec.GetAttributeAtPath(paths[i] + ".xformOpOrder")
                if op is None:
                    op = Sdf.AttributeSpec(spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
                op.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
    Box.size_table = {"dims": dims, "level": width_level(dims[:, 1], w_start, w_step, max_level),
                      "ref": ref if fixed_refs else None, "is_sphere": is_sphere}
    Box.max_level = max_level
    print(f"[holdv4] object sizes: {n} envs, lateral {dims[:, 1].min():.2f}-{dims[:, 1].max():.2f} m, "
          f"{int(is_sphere.sum())} spheres, fixed_refs={fixed_refs}")
