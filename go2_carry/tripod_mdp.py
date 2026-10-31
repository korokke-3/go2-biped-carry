"""MDP terms for the Tripod task (2026-10-26): Go2 on three legs (FL, RL, RR) with the right front leg (FR) lifted.

Two gait modes in one policy, as go2_quad2hand's gait_mode (one-hot observation, per-mode period / offsets):
  mode 0 "hop"  (けんけん): the front leg alternates with the two hind legs moving together, as three-legged dogs hop.
  mode 1 "walk": three-beat walk (one leg swings at a time), optionally with a bag hung on the lifted forearm.
At zero command both modes stand still on the three feet (user, 2026-10-26: a tripod is statically stable, no
stepping in place - unlike the reference's always-on clock).

The bag is a point mass on a rope hung at the FR elbow (knee joint), integrated here (position-based rope constraint)
and applied to the FR calf as the rope tension. It stays on the forearm only while the forearm slopes up from the
elbow to the paw (foot above knee); otherwise the handle slides off the paw.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, sample_uniform

from . import refwalk_mdp as rmdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

STANCE_FEET = ["FL_foot", "RL_foot", "RR_foot"]
FREE_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"]
# lifted FR pose: thigh -1.4 (upper arm forward, elbow ~ at shoulder height), calf -1.2 -> forearm 59 deg up from the
# elbow (planar kinematics, thigh = calf = 0.213 m): the paw is above the elbow, a handle slides to the elbow and stays
LIFT_POSE = [0.0, -1.4, -1.2]
# per mode: period [s], stance fraction, offsets (FL, RL, RR)
GAIT = {
    0: (0.4, 0.6, [0.0, 0.5, 0.5]),          # hop: ref hand-mode period 0.4 / stance 0.6; hinds together vs front
    1: (0.6, 0.75, [0.0, 1 / 3, 2 / 3]),     # walk: 3-beat, swing 0.25 each -> never two legs in the air
}


def stance_feet_for(free: str) -> list[str]:
    """The three stance feet with the lifted leg removed, in gait-offset order: the hop pairs the first foot against
    the other two, so the lone foot is the one on the lifted leg's end of the body (FR lifted: FL vs RL+RR; RR lifted:
    RL vs FL+FR)."""
    if free == "FR":
        return list(STANCE_FEET)
    if free == "RR":
        return ["RL_foot", "FL_foot", "FR_foot"]
    raise ValueError(free)


class Tri:
    mode: torch.Tensor | None = None       # (N,) long
    period: torch.Tensor | None = None     # (N,)
    stance: torch.Tensor | None = None     # (N,)
    offsets: torch.Tensor | None = None    # (N, 3)
    # bag
    present: torch.Tensor | None = None    # (N,) bool: the episode gets a bag
    hung: torch.Tensor | None = None       # (N,) bool: currently hanging on the forearm
    t_hang: torch.Tensor | None = None     # (N,) s
    mass: torch.Tensor | None = None       # (N,)
    rope: torch.Tensor | None = None       # (N,)
    pos: torch.Tensor | None = None        # (N, 3) world
    vel: torch.Tensor | None = None        # (N, 3)
    tension: torch.Tensor | None = None    # (N, 3) force on the elbow (world)
    elbow_prev: torch.Tensor | None = None # (N, 3) elbow at the previous policy step (pivot interpolated in between)
    slip_t: torch.Tensor | None = None     # (N,) s the forearm has sloped down
    lost: torch.Tensor | None = None       # (N,) bool (termination flag of this step)
    fixed_mode: int | None = None          # evaluation overrides
    fixed_bag: float | None = None         # kg (0 = no bag)
    # lifted leg (TripodHind, 2026-10-28): set by reset_tripod(free_leg=...); FR for Tripod / High / Cool
    free: str = "FR"
    stance_feet: list[str] = STANCE_FEET
    free_joints: list[str] = FREE_JOINTS


def _init(env):
    n, dev = env.num_envs, env.device
    if Tri.mode is None or Tri.mode.shape[0] != n:
        Tri.mode = torch.zeros(n, dtype=torch.long, device=dev)
        Tri.period = torch.full((n,), GAIT[0][0], device=dev)
        Tri.stance = torch.full((n,), GAIT[0][1], device=dev)
        Tri.offsets = torch.tensor(GAIT[0][2], device=dev).repeat(n, 1)
        Tri.present = torch.zeros(n, dtype=torch.bool, device=dev)
        Tri.hung = torch.zeros(n, dtype=torch.bool, device=dev)
        Tri.t_hang = torch.zeros(n, device=dev)
        Tri.mass = torch.zeros(n, device=dev)
        Tri.rope = torch.full((n,), 0.15, device=dev)
        Tri.pos = torch.zeros(n, 3, device=dev)
        Tri.vel = torch.zeros(n, 3, device=dev)
        Tri.tension = torch.zeros(n, 3, device=dev)
        Tri.elbow_prev = torch.zeros(n, 3, device=dev)
        Tri.slip_t = torch.zeros(n, device=dev)
        Tri.lost = torch.zeros(n, dtype=torch.bool, device=dev)


# ----------------------------------------------------------------------------- reset
def reset_tripod(env: ManagerBasedRLEnv, env_ids: torch.Tensor, p_walk: float, p_bag: float,
                 mass_range: tuple[float, float], rope_range: tuple[float, float], hang_range_s: tuple[float, float],
                 p_lifted: float, free_leg: str = "FR", lift_pose: list[float] | None = None,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Reset event (after the quadruped reset): gait mode, bag, and for a share of the episodes the FR leg already
    lifted (reference-state init, cf. HoldRef's tray pose)."""
    _init(env)
    Tri.free, Tri.stance_feet = free_leg, stance_feet_for(free_leg)
    Tri.free_joints = [f"{free_leg}_hip_joint", f"{free_leg}_thigh_joint", f"{free_leg}_calf_joint"]
    dev, k = env.device, len(env_ids)
    mode = (torch.rand(k, device=dev) < p_walk).long()
    if Tri.fixed_mode is not None:
        mode[:] = Tri.fixed_mode
    Tri.mode[env_ids] = mode
    per = torch.tensor([GAIT[0][0], GAIT[1][0]], device=dev)
    sta = torch.tensor([GAIT[0][1], GAIT[1][1]], device=dev)
    off = torch.tensor([GAIT[0][2], GAIT[1][2]], device=dev)
    Tri.period[env_ids], Tri.stance[env_ids], Tri.offsets[env_ids] = per[mode], sta[mode], off[mode]
    bag = (mode == 1) & (torch.rand(k, device=dev) < p_bag)
    m = sample_uniform(*mass_range, (k,), dev)
    if Tri.fixed_bag is not None:
        bag[:] = Tri.fixed_bag > 0
        m[:] = Tri.fixed_bag
    Tri.present[env_ids] = bag
    Tri.mass[env_ids] = m
    Tri.rope[env_ids] = sample_uniform(*rope_range, (k,), dev)
    Tri.t_hang[env_ids] = sample_uniform(*hang_range_s, (k,), dev)
    Tri.hung[env_ids] = False
    Tri.slip_t[env_ids] = 0.0
    Tri.tension[env_ids] = 0.0
    Tri.lost[env_ids] = False
    lift = torch.rand(k, device=dev) < p_lifted
    if lift.any():
        robot: Articulation = env.scene[asset_cfg.name]
        ids = env_ids[lift]
        jid = robot.find_joints(Tri.free_joints, preserve_order=True)[0]
        q = robot.data.joint_pos[ids].clone()
        qd = robot.data.joint_vel[ids].clone()
        q[:, jid] = torch.tensor(LIFT_POSE if lift_pose is None else lift_pose, device=dev) + sample_uniform(-0.1, 0.1, (len(ids), len(jid)), dev)
        qd[:, jid] = 0.0
        robot.write_joint_state_to_sim(q, qd, env_ids=ids)


# ----------------------------------------------------------------------------- observations
def gait_mode_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    _init(env)
    return torch.nn.functional.one_hot(Tri.mode, 2).float()


def gait_clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """sin / cos of the gait phase with the env's own period (unitree_rl_lab gait_phase, per-mode period)."""
    _init(env)
    ph = (env.episode_length_buf * env.step_dt) % Tri.period / Tri.period
    return torch.stack([torch.sin(2 * torch.pi * ph), torch.cos(2 * torch.pi * ph)], dim=1)


def bag_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Critic only: hung flag, mass, rope, bag position / velocity relative to the elbow in the base frame."""
    _init(env)
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.root_quat_w
    on = Tri.hung.float()[:, None]
    rel = quat_apply_inverse(q, Tri.pos - _elbow(env)) * on
    vel = quat_apply_inverse(q, Tri.vel - robot.data.root_lin_vel_w) * on
    return torch.cat([on, Tri.mass[:, None] * on, Tri.rope[:, None] * on, rel, vel * 0.5], dim=1)


# ----------------------------------------------------------------------------- helpers
def _moving(env) -> torch.Tensor:
    cmd = env.command_manager.get_command("base_velocity")
    return cmd.abs().sum(dim=1) > 0.0


def _elbow(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_pos_w[:, robot.find_bodies(f"{Tri.free}_calf")[0][0]]


def _paw(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_pos_w[:, robot.find_bodies(f"{Tri.free}_foot")[0][0]]


# ----------------------------------------------------------------------------- rewards
def tripod_gait(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """go2_quad2hand _reward_gait_quad over the three stance feet with the mode's clock while a command is given;
    at zero command: all three feet down (stand still, no stepping in place)."""
    _init(env)
    ph = (env.episode_length_buf * env.step_dt) % Tri.period / Tri.period
    leg = (ph[:, None] + Tri.offsets) % 1.0
    c = rmdp._contact_filt(env, sensor_cfg)
    walk = (c == (leg < Tri.stance[:, None])).float().mean(dim=1)
    stand = c.float().mean(dim=1)
    return torch.where(_moving(env), walk, stand)


def base_orientation_flat(env: ManagerBasedRLEnv, sigma: float, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """_reward_body_orientation_quad (pitch target 0): exp(-(roll^2 + pitch^2) / sigma), via projected gravity."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.exp(-(torch.square(torch.asin(g[:, 1].clamp(-1, 1))) + torch.square(torch.asin(g[:, 0].clamp(-1, 1))))
                     / sigma)


def com_over_support(env: ManagerBasedRLEnv, std: float, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """Standing only: horizontal distance of the CoM (robot + hanging bag) from the centroid of the three stance feet,
    exp kernel (ALaM, Shin et al. 2025: minimise the gravitational moment about the support centroid)."""
    _init(env)
    robot: Articulation = env.scene[asset_cfg.name]
    m = robot.root_physx_view.get_masses().to(env.device)
    xy = (m[..., None] * robot.data.body_com_pos_w[..., :2]).sum(dim=1)
    bm = (Tri.mass * Tri.hung.float())[:, None]
    xy = (xy + bm * Tri.pos[:, :2]) / (m.sum(dim=1, keepdim=True) + bm)
    feet = robot.data.body_pos_w[:, robot.find_bodies(Tri.stance_feet, preserve_order=True)[0], :2].mean(dim=1)
    r = torch.exp(-torch.sum(torch.square(xy - feet), dim=1) / std**2)
    return r * (~_moving(env)).float()


def forearm_up(env: ManagerBasedRLEnv, scale: float = 0.1) -> torch.Tensor:
    """Paw above the elbow (forearm sloping up, a handle stays on), clipped to [-1, 1] per `scale` m. Chunk 1: the hop
    mode kept the paw 15 cm above the elbow, the walk mode 8 cm below, and every bag slid off; the dof prior to
    LIFT_POSE (-0.1) alone did not lift it."""
    return ((_paw(env)[:, 2] - _elbow(env)[:, 2]) / scale).clamp(-1.0, 1.0)


def bag_hung(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 per step while the bag hangs on the forearm (losing it ends the episode)."""
    _init(env)
    return Tri.hung.float()


# ----------------------------------------------------------------------------- bag dynamics (interval event)
def bag_step(env: ManagerBasedRLEnv, env_ids, substeps: int = 4, slip_s: float = 0.2, floor: float = 0.04,
             damping: float = 0.5, max_g: float = 6.0, stretch: float = 0.01, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """Every policy step: hang the bag (at t_hang, once the paw is above the elbow, or 3 s later regardless - a person
    hangs it), integrate the rope pendulum about the elbow and apply the rope tension to the FR calf. The bag slides
    off when the paw stays below the elbow for slip_s.
    The elbow is interpolated linearly from the previous policy step over the substeps: moving it in one substep
    gave 0.2-0.3 kN tension spikes for a 2 kg bag (scripts/diagnostics/diag_tripod.py); the tension is capped at max_g x m g."""
    _init(env)
    robot: Articulation = env.scene[asset_cfg.name]
    dev = env.device
    t = env.episode_length_buf.float() * env.step_dt
    elbow, paw = _elbow(env), _paw(env)
    up = paw[:, 2] > elbow[:, 2]
    new = Tri.present & ~Tri.hung & ~Tri.lost & (t >= Tri.t_hang) & (up | (t >= Tri.t_hang + 3.0))
    fresh = env.episode_length_buf <= 1
    Tri.elbow_prev[fresh] = elbow[fresh]
    if new.any():
        Tri.hung[new] = True
        Tri.pos[new] = elbow[new] - torch.tensor([0.0, 0.0, 1.0], device=dev) * Tri.rope[new, None]
        Tri.vel[new] = robot.data.root_lin_vel_w[new]
        Tri.slip_t[new] = 0.0
    Tri.slip_t = torch.where(Tri.hung & ~up, Tri.slip_t + env.step_dt, torch.zeros_like(Tri.slip_t))
    Tri.lost = Tri.hung & (Tri.slip_t > slip_s)
    Tri.hung = Tri.hung & ~Tri.lost
    # bag pendulum on the strap
    h = env.step_dt / substeps
    g = torch.tensor([0.0, 0.0, -9.81], device=dev)
    z0 = env.scene.env_origins[:, 2] + floor
    # strap = stiff spring-damper that only pulls (1 cm stretch under m g, critically damped): an inextensible
    # rope (position projection) let the tension flip between slack and taut every policy step (11 / 20 N for 2 kg)
    k = Tri.mass * 9.81 / stretch
    c = 2.0 * torch.sqrt(k * Tri.mass)
    f_robot = torch.zeros(env.num_envs, 3, device=dev)
    for i in range(substeps):
        a = (i + 1) / substeps
        piv = Tri.elbow_prev + (elbow - Tri.elbow_prev) * a
        piv_v = (elbow - Tri.elbow_prev) / env.step_dt
        d = Tri.pos - piv
        n = d.norm(dim=1, keepdim=True).clamp(min=1e-6)
        u = d / n
        ext = (n[:, 0] - Tri.rope).clamp(min=0.0)
        vr = ((Tri.vel - piv_v) * u).sum(dim=1)
        t_mag = (k * ext + c * vr * (ext > 0).float()).clamp(min=0.0)  # pulls only
        f_bag = -t_mag[:, None] * u  # on the bag, towards the elbow
        f_robot = f_robot - f_bag
        Tri.vel = Tri.vel * (1.0 - damping * h) + (g + f_bag / Tri.mass.clamp(min=1e-3)[:, None]) * h
        Tri.pos = Tri.pos + Tri.vel * h
        below = Tri.pos[:, 2] < z0  # the floor (reaction to the ground, not to the robot)
        Tri.pos[:, 2] = torch.where(below, z0, Tri.pos[:, 2])
        Tri.vel[:, 2] = torch.where(below, Tri.vel[:, 2].clamp(min=0.0), Tri.vel[:, 2])
    Tri.elbow_prev = elbow.clone()
    f = f_robot / substeps
    cap = (max_g * 9.81 * Tri.mass)[:, None]
    f = f * (cap / f.norm(dim=1, keepdim=True).clamp(min=1e-6)).clamp(max=1.0)
    Tri.tension = f * Tri.hung.float()[:, None]
    Tri.pos = torch.where(Tri.hung[:, None], Tri.pos, elbow)
    Tri.vel = torch.where(Tri.hung[:, None], Tri.vel, robot.data.root_lin_vel_w)
    # apply at the elbow: force on the calf + the moment of acting at the joint instead of the calf CoM
    bid = robot.find_bodies(f"{Tri.free}_calf")[0]
    f = Tri.tension[:, None, :]
    r = (elbow - robot.data.body_com_pos_w[:, bid[0]])[:, None, :]
    robot.set_external_force_and_torque(f, torch.cross(r, f, dim=-1), body_ids=bid, is_global=True)


def bag_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    _init(env)
    return Tri.lost.clone()


# ----------------------------------------------------------------------------- TripodHigh (2026-10-28)
def _walk(env) -> torch.Tensor:
    _init(env)
    return (Tri.mode == 1).float()


def base_pose_by_mode(env: ManagerBasedRLEnv, pitch: list[float], height: list[float], sigma_pitch: float,
                      sigma_height: float, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """Per-mode body pose: exp(-(roll^2 + (pitch_up - target)^2) / sigma) * exp(-(h - target)^2 / sigma_h); pitch_up > 0
    = nose up (go2_quad2hand body_orientation / base_height terms with a mode-dependent target)."""
    _init(env)
    robot: Articulation = env.scene[asset_cfg.name]
    g = robot.data.projected_gravity_b
    roll = torch.asin(g[:, 1].clamp(-1, 1))
    up = torch.asin((-g[:, 0]).clamp(-1, 1))
    p_t = torch.tensor(pitch, device=env.device)[Tri.mode]
    h_t = torch.tensor(height, device=env.device)[Tri.mode]
    h = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (torch.exp(-(roll**2 + (up - p_t) ** 2) / sigma_pitch) * torch.exp(-(h - h_t) ** 2 / sigma_height))


def elbow_height(env: ManagerBasedRLEnv, low: float, high: float) -> torch.Tensor:
    """Walk mode: elbow (where the bag hangs) height above the floor, linear 0 at `low` -> 1 at `high`."""
    z = _elbow(env)[:, 2] - env.scene.env_origins[:, 2]
    return ((z - low) / (high - low)).clamp(0.0, 1.0) * _walk(env)


# ----------------------------------------------------------------------------- motor heat, all joints (2026-10-28)
def _peak_nm(env, robot):
    if not hasattr(env, "_tri_peak_nm"):  # ref URDF effort limits: hip / thigh 23.7, calf 35.55 (knee x1.5)
        env._tri_peak_nm = torch.tensor([35.55 if "calf" in n else 23.7 for n in robot.joint_names], device=env.device)
    return env._tri_peak_nm


def motor_heat(env, asset_cfg=SceneEntityCfg("robot")):
    """Copper loss ~ torque^2: sum over all 12 motors of (applied torque / peak)^2 (RefWalk's hind_torque_load, all legs).
    TripodHigh c2 held the lifted FR thigh on its stop at 20.5 Nm RMS (0.86 of peak) with no bag, and locked the FL knee
    straight on its stop at 12-14 Nm."""
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.applied_torque / _peak_nm(env, robot)), dim=1)


def motor_heat_max(env, asset_cfg=SceneEntityCfg("robot")):
    """The hottest motor trips the thermal protection first: max over the motors of (applied torque / peak)^2."""
    robot = env.scene[asset_cfg.name]
    return torch.square(robot.data.applied_torque / _peak_nm(env, robot)).max(dim=1)[0]
