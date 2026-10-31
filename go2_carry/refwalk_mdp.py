"""MDP terms for the RefWalk task: go2_quad2hand's biped ("hand") mode, mirrored to the hind legs.

go2_quad2hand (refs/go2_quad2hand, go2_rl/legged_gym/envs/go2/go2_quad2hand.py) walks on the FRONT legs with the
body pitched 84 deg nose-down; here the stance legs are the hind legs and the free legs are the front legs.
Each function names the reference reward it reproduces.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FOOT_HEIGHT_OFFSET = 0.022  # go2_quad2hand foot_height_offset: foot frame origin above the ground [m]


# ----------------------------------------------------------------------------- commands
class DeadzoneVelocityCommand(UniformVelocityCommand):
    """go2_quad2hand _resample_commands: uniform sampling, then each component with |c| <= 0.1 is set to 0
    (gives standing / pure-turning / pure-walking commands without a separate standing fraction).
    With ``base_ranges`` set, a fraction ``p_base`` of the resamples is drawn from those (the reference range) and
    the rest from ``ranges`` (grown by :func:`cmd_range_curriculum`), so the common speeds keep being practised."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        b = self.cfg.base_ranges
        if b is not None and self.cfg.p_base > 0.0:
            ids = torch.as_tensor(env_ids, device=self.device)
            pick = ids[torch.rand(len(ids), device=self.device) < self.cfg.p_base]
            if len(pick):
                r = torch.empty(len(pick), device=self.device)
                self.vel_command_b[pick, 0] = r.uniform_(*b.lin_vel_x)
                self.vel_command_b[pick, 1] = r.uniform_(*b.lin_vel_y)
                self.vel_command_b[pick, 2] = r.uniform_(*b.ang_vel_z)
        c = self.vel_command_b[env_ids]
        self.vel_command_b[env_ids] = c * (c.abs() > self.cfg.deadzone)


@configclass
class DeadzoneVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = DeadzoneVelocityCommand
    deadzone: float = 0.1
    base_ranges: UniformVelocityCommandCfg.Ranges | None = None
    p_base: float = 0.0
    limit_ranges: UniformVelocityCommandCfg.Ranges | None = None


def cmd_range_curriculum(env: ManagerBasedRLEnv, env_ids, lin_term: str, ang_term: str, lin_threshold: float,
                         ang_threshold: float, step: float = 0.1):
    """Grow the command ranges axis by axis up to ``limit_ranges`` while tracking is good (Go2's lin/ang_vel_cmd_levels
    with per-axis limits and an own threshold for yaw). Every episode length; state persists across resumes via
    $REFWALK_CMD_START (json {"x": [lo, hi], "y": [...], "z": [...]}) and is appended to $REFWALK_CMD_STATUS."""
    import json
    import os
    term = env.command_manager.get_term("base_velocity")
    r, lim = term.cfg.ranges, term.cfg.limit_ranges
    if not getattr(env, "_cmd_curr_started", False):
        start = os.environ.get("REFWALK_CMD_START")
        if start:
            st = json.loads(start)
            r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = tuple(st["x"]), tuple(st["y"]), tuple(st["z"])
        env._cmd_curr_started = True
    # once per episode length (Go2's "counter % length == 0" needs a reset exactly at that step; with few envs it can miss)
    last = getattr(env, "_cmd_curr_last", 0)
    if env.common_step_counter - last >= env.max_episode_length:
        env._cmd_curr_last = env.common_step_counter
        sums = env.reward_manager._episode_sums
        lw = env.reward_manager.get_term_cfg(lin_term).weight
        aw = env.reward_manager.get_term_cfg(ang_term).weight
        lin = float(torch.mean(sums[lin_term][env_ids]) / env.max_episode_length_s)
        ang = float(torch.mean(sums[ang_term][env_ids]) / env.max_episode_length_s)

        def grow(cur, lo_lim, hi_lim):
            return (max(lo_lim, round(cur[0] - step, 3)), min(hi_lim, round(cur[1] + step, 3)))
        if lin > lw * lin_threshold:
            r.lin_vel_x = grow(r.lin_vel_x, lim.lin_vel_x[0], lim.lin_vel_x[1])
            r.lin_vel_y = grow(r.lin_vel_y, lim.lin_vel_y[0], lim.lin_vel_y[1])
        if ang > aw * ang_threshold:
            r.ang_vel_z = grow(r.ang_vel_z, lim.ang_vel_z[0], lim.ang_vel_z[1])
        f = os.environ.get("REFWALK_CMD_STATUS")
        if f:
            with open(f, "a") as fh:
                fh.write(json.dumps({"step": int(env.common_step_counter), "lin": round(lin / lw, 3), "ang": round(ang / aw, 3),
                                     "x": list(r.lin_vel_x), "y": list(r.lin_vel_y), "z": list(r.ang_vel_z)}) + "\n")
    return torch.tensor(r.lin_vel_x[1], device=env.device)


# ----------------------------------------------------------------------------- helpers
def _contact_filt(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """legged_gym contact_filt: contact (force > 1 N) now OR at the previous policy step, for the selected bodies."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0
    key = "_refwalk_contact_" + "_".join(str(i) for i in sensor_cfg.body_ids)
    cache = getattr(env, key, None)
    step = env.common_step_counter
    if cache is None or cache[0] != step:
        last = cache[2] if cache is not None else torch.zeros_like(contact)
        if cache is not None and cache[0] == step - 1:
            last = cache[1]
        last = last & (env.episode_length_buf > 1)[:, None]  # reset_idx clears last_contacts
        cache = (step, contact, last)
        setattr(env, key, cache)
    return cache[1] | cache[2]


def biped_lean_roll(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(forward lean, sideways lean) [rad] of the upright biped. Forward lean > 0 = head towards the belly
    (= walking direction). In go2_quad2hand's handstand, pitch 84 deg = 6 deg short of vertical with the top of the
    body displaced towards the belly, i.e. forward lean +6 deg here."""
    asset: Articulation = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b  # upright: (-1, 0, 0)
    lean = torch.atan2(-g[:, 2], -g[:, 0])
    roll = torch.asin(g[:, 1].clamp(-1.0, 1.0))
    return torch.stack([lean, roll], dim=1)


# ----------------------------------------------------------------------------- rewards
def body_orientation_hand(env: ManagerBasedRLEnv, lean_target: float, sigma: float) -> torch.Tensor:
    """_reward_body_orientation_hand: exp(-(roll^2 + (pitch - target)^2) / sigma)."""
    lr = biped_lean_roll(env)
    return torch.exp(-(torch.square(lr[:, 1]) + torch.square(lr[:, 0] - lean_target)) / sigma)


def base_height_hand(env: ManagerBasedRLEnv, target: float, sigma: float,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """_reward_base_height_hand: exp(-(h - target)^2 / sigma)."""
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-torch.square(h - target) / sigma)


def gait_hand(env: ManagerBasedRLEnv, period: float, offset: list[float], stance_frac: float,
              sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """_reward_gait_hand: mean over the stance legs of [contact_filt == (leg phase < stance_frac)], always on
    (also at zero command: the reference steps in place)."""
    phase = (env.episode_length_buf * env.step_dt) % period / period
    leg_phase = (phase[:, None] + torch.tensor(offset, device=env.device)) % 1.0
    match = _contact_filt(env, sensor_cfg) == (leg_phase < stance_frac)
    return match.float().mean(dim=1)


def foot_clearance_hand(env: ManagerBasedRLEnv, target: float, sigma: float,
                        asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """_reward_foot_clearance_hand: exp(-sum_feet |v_xy| (z - target - offset)^2 / sigma) over the stance legs."""
    asset: Articulation = env.scene[asset_cfg.name]
    v_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    err = torch.sum(v_xy * torch.square(z - target - FOOT_HEIGHT_OFFSET), dim=1)
    return torch.exp(-err / sigma)


def contact_no_vel(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """_reward_contact_no_vel: sum over feet in contact of |v_foot|^2 (sensor_cfg / asset_cfg list the same feet in
    the same order)."""
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    c = _contact_filt(env, sensor_cfg).float()
    return torch.sum(torch.square(v) * c[:, :, None], dim=(1, 2))


def joint_dev_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target: list[float] | None = None) -> torch.Tensor:
    """_reward_hip_close_to_default / _reward_rear_dof_pos_hand: sum of squared joint deviations from the default
    pose (target=None) or from an explicit target (in asset_cfg.joint_names order, preserve_order=True)."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    ref = asset.data.default_joint_pos[:, asset_cfg.joint_ids] if target is None else torch.tensor(target, device=env.device)
    return torch.sum(torch.square(q - ref), dim=1)


# ----------------------------------------------------------------------------- privileged observations
def _actuator_param(env, asset_cfg, name: str) -> torch.Tensor:
    """(N, num_joints) actuator gain over all actuator groups, in articulation joint order."""
    asset: Articulation = env.scene[asset_cfg.name]
    out = torch.zeros(env.num_envs, asset.num_joints, device=env.device)
    for act in asset.actuators.values():
        out[:, act.joint_indices] = getattr(act, name)
    return out


def actuator_kp_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Per-joint Kp / nominal Kp (go2_quad2hand privileged p_gain_scales)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _actuator_param(env, asset_cfg, "stiffness") / asset.data.default_joint_stiffness.clamp(min=1e-6)


def actuator_kd_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return _actuator_param(env, asset_cfg, "damping") / asset.data.default_joint_damping.clamp(min=1e-6)


def joint_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """go2_quad2hand privileged joint_frictions (scaled by 20 to ~[0.1, 1])."""
    return env.scene[asset_cfg.name].data.joint_friction_coeff * 20.0


def joint_armature(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """go2_quad2hand privileged joint_armatures (scaled by 20 to ~[0.2, 1])."""
    return env.scene[asset_cfg.name].data.joint_armature * 20.0


# ----------------------------------------------------------------------------- events (USD, before the sim starts)
def randomize_foot_torsional_friction(env, env_ids, radius_range: tuple[float, float],
                                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                      foot_names: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")):
    """PhysX torsional patch radius of the foot colliders, per env (mode "usd", replicate_physics=False).
    MuJoCo's Go2 feet (condim 4/6: torsional 0.02 m) cut RefWalk-v2's turn rate to 55 % of the command; PhysX spheres
    have no torsional friction by default (radius 0). Rolling friction (MuJoCo condim 6) has no PhysX counterpart."""
    from pxr import PhysxSchema, Usd, UsdPhysics

    from isaacsim.core.utils.stage import get_current_stage

    import isaaclab.sim as sim_utils

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    roots = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    ids = range(len(roots)) if env_ids is None else env_ids.cpu().tolist()
    r = torch.empty(len(roots)).uniform_(*radius_range)
    n_set = 0
    for i in ids:
        for foot in foot_names:
            prim = stage.GetPrimAtPath(f"{roots[i]}/{foot}")
            if not prim.IsValid():
                continue
            # the URDF importer makes the collider subtrees instanceable (shared prototypes): un-instance the foot's
            for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if p.IsInstance():
                    p.SetInstanceable(False)
            for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if not p.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if p.IsInstanceProxy():
                    raise RuntimeError(f"foot collider {p.GetPath()} is an instance proxy: cannot randomize per env")
                api = PhysxSchema.PhysxCollisionAPI.Apply(p)
                api.CreateTorsionalPatchRadiusAttr().Set(float(r[i]))
                api.CreateMinTorsionalPatchRadiusAttr().Set(float(r[i]))
                n_set += 1
    print(f"[refwalk] torsional patch radius on {n_set} foot colliders, range {radius_range}")
    env._foot_torsional_radius = r


# ----------------------------------------------------------------------------- hind-leg motor load (2026-10-22)
HIND_PEAK_NM = {"RL_hip_joint": 23.7, "RR_hip_joint": 23.7, "RL_thigh_joint": 23.7, "RR_thigh_joint": 23.7,
                "RL_calf_joint": 35.55, "RR_calf_joint": 35.55}  # ref URDF effort limits
# $LOAD_STATUS_FILE: a separate status file per experiment line (HoldBall, 2026-10-24)
LOAD_STATUS = os.path.expanduser(os.environ.get("LOAD_STATUS_FILE", "~/go2_sim2real/runs/load_penalty.status"))


def hind_torque_load(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Motor heating ~ torque^2: sum over the hind joints of (applied torque / peak torque)^2, while upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_hind_load_ids"):
        names = list(HIND_PEAK_NM)
        env._hind_load_ids = torch.tensor([asset.joint_names.index(n) for n in names], device=env.device)
        env._hind_load_peak = torch.tensor([HIND_PEAK_NM[n] for n in names], device=env.device)
    tau = asset.data.applied_torque[:, env._hind_load_ids] / env._hind_load_peak
    upright = (-asset.data.projected_gravity_b[:, 0] > 0.8).float()
    return torch.sum(torch.square(tau), dim=1) * upright


def load_penalty_curriculum(env: ManagerBasedRLEnv, env_ids, term_name: str, lin_term: str, step: float = 0.1,
                            w_max: float = 2.0, up_track: float = 0.8, down_track: float = 0.7,
                            up_fall: float = 0.05, down_fall: float = 0.10):
    """Raise the (negative) weight of ``term_name`` by ``step`` per episode length while tracking >= up_track x weight
    and the fall rate < up_fall; lower it when tracking < down_track or falls > down_fall. Starts from the last line of
    LOAD_STATUS (persists across chunked runs), weight 0 otherwise."""
    import json
    rm = env.reward_manager
    cfg = rm.get_term_cfg(term_name)
    if not getattr(env, "_load_curr_started", False):
        if os.path.exists(LOAD_STATUS):
            with open(LOAD_STATUS) as fh:
                lines = fh.read().splitlines()
            if lines:
                cfg.weight = float(json.loads(lines[-1])["weight"])
        env._load_curr_started = True
        env._load_fall_n = 0
        env._load_reset_n = 0
    tm = env.termination_manager
    if len(env_ids):
        env._load_reset_n += len(env_ids)
        # robot falls only (in the hold tasks, dropping the object also terminates)
        fell = tm.get_term("body_contact") if "body_contact" in tm.active_terms else tm.terminated
        env._load_fall_n += int(fell[env_ids].sum())
    last = getattr(env, "_load_curr_last", 0)
    if env.common_step_counter - last >= env.max_episode_length:
        env._load_curr_last = env.common_step_counter
        lw = rm.get_term_cfg(lin_term).weight
        # per-second average over each ended episode's ACTUAL length: dividing by the max length (Go2's
        # lin_vel_cmd_levels) reads short episodes as poor tracking; in the hold tasks a dropped object ends the
        # episode, and the weight fell 1.0 -> 0.9 in HoldV4 chunk 5 with a ratio of 0.16 while the gait was unchanged
        if len(env_ids):
            dur = (env.episode_length_buf[env_ids].float() * env.step_dt).clamp(min=env.step_dt)
            lin = float(torch.mean(rm._episode_sums[lin_term][env_ids] / dur)) / lw
        else:
            lin = 0.0
        fall = env._load_fall_n / max(env._load_reset_n, 1)
        w = -cfg.weight
        if lin >= up_track and fall < up_fall:
            w = min(w_max, w + step)
        elif lin < down_track or fall > down_fall:
            w = max(0.0, w - step)
        cfg.weight = -round(w, 3)
        env._load_fall_n = 0
        env._load_reset_n = 0
        with open(LOAD_STATUS, "a") as fh:
            fh.write(json.dumps({"step": int(env.common_step_counter), "weight": cfg.weight, "lin": round(lin, 3),
                                 "fall": round(fall, 3)}) + "\n")
    return torch.tensor(-cfg.weight, device=env.device)


def joint_limit_margin(env: ManagerBasedRLEnv, margin: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum over the selected joints of how far they are inside the last ``margin`` rad before a hard limit.
    The motor-load penalty alone was gamed (RefWalk-V4 chunk 2): the knees were locked straight on the mechanical stop
    (calf -0.837 = limit), carrying the weight with zero motor torque; that fell in MuJoCo and would hammer the real
    stops. Isaac Lab's joint_pos_limits only acts beyond the (soft) limit, i.e. not on a joint resting exactly on it."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lim = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    lo = (lim[..., 0] + margin - q).clamp(min=0.0)
    hi = (q - (lim[..., 1] - margin)).clamp(min=0.0)
    return torch.sum(lo + hi, dim=1)


# ----------------------------------------------------------------------------- motor strength (2026-10-25)
def randomize_motor_strength(env: ManagerBasedRLEnv, env_ids, scale_range: tuple[float, float],
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset event: per env, scale the torque-speed curve of every Unitree actuator (peak torques Y1 / Y2 and the
    speeds X1 / X2) by one factor ~ U(scale_range). HoldV4 c11+ and HoldBall c6 stood up in MuJoCo only with the
    training curve (--unitree_curve): with MuJoCo's flat limits (knee 45.4 Nm vs the curve's 30-35 Nm) they
    overshot and fell; a flat 30 Nm limit did not fall. The real motors' strength around the curve is an estimate
    (knee x1.5), so the policy should not depend on it."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    s = torch.empty(len(env_ids), 1, device=env.device).uniform_(*scale_range)
    for act in asset.actuators.values():
        if not hasattr(act, "_effort_y1"):
            continue
        if not hasattr(act, "_base_curve"):
            act._base_curve = [t.clone() for t in (act._effort_y1, act._effort_y2, act._velocity_x1, act._velocity_x2)]
            print(f"[refwalk] motor strength x{scale_range} on {type(act).__name__}, Y1 {tuple(act._effort_y1.shape)} "
                  f"{act._effort_y1[0].tolist()}", flush=True)
        for t, base in zip((act._effort_y1, act._effort_y2, act._velocity_x1, act._velocity_x2), act._base_curve):
            t[env_ids] = base[env_ids] * s
