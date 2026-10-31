"""Custom MDP terms for Go2 bipedal (hind-leg) standing / walking / carrying.

Conventions
-----------
* Go2 base frame: x forward, y left, z up (in the normal quadruped pose).
* "Upright" biped pose = body pitched nose-up by 90 deg, so body +x points to the sky and the
  gravity vector in the body frame is (-1, 0, 0).
* In that pose the belly (body -z) faces the walking direction; we call its horizontal
  projection the *heading* of the biped. Velocity commands (vx, vy, wz) are expressed in this
  heading frame (vx: towards the belly, vy: to the robot's left, wz: yaw rate about world z).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

UPRIGHT_GRAVITY_B = (-1.0, 0.0, 0.0)

# Upright initial pose: hind legs roughly under the hip, front legs folded (radians).
UPRIGHT_JOINT_POS = {
    "R[L,R]_hip_joint": 0.0,
    "R[L,R]_thigh_joint": 2.1,
    "R[L,R]_calf_joint": -1.3,
    "F[L,R]_hip_joint": 0.0,
    "F[L,R]_thigh_joint": 0.0,
    "F[L,R]_calf_joint": -2.0,
}
UPRIGHT_BASE_HEIGHT = 0.56


# ----------------------------------------------------------------------------- helpers
def uprightness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """cos of the angle between body +x and world up (1 = perfectly upright biped)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return -asset.data.projected_gravity_b[:, 0]


def biped_heading_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Unit horizontal vector the belly faces (world frame), shape (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    belly = quat_apply(asset.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3))
    h = belly[:, :2]
    return h / h.norm(dim=1, keepdim=True).clamp(min=1e-6)


def biped_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base linear velocity in the biped heading frame: (forward=belly, left), shape (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    h = biped_heading_xy(env, asset_cfg)
    left = torch.stack([-h[:, 1], h[:, 0]], dim=1)
    v = asset.data.root_lin_vel_w[:, :2]
    return torch.stack([(v * h).sum(1), (v * left).sum(1)], dim=1)


# ----------------------------------------------------------------------------- observations
def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]


def biped_lin_vel_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return biped_lin_vel(env, asset_cfg)


# ----------------------------------------------------------------------------- rewards
def upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Smooth reward in [0, 1]: ((1 + cos) / 2)^2 with cos = alignment of body +x with world up."""
    return torch.square(0.5 * uprightness(env, asset_cfg) + 0.5)


def height_reward(env: ManagerBasedRLEnv, low: float, high: float,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear ramp in [0, 1] from `low` to `high` base height."""
    z = base_height(env, asset_cfg)[:, 0]
    return ((z - low) / (high - low)).clamp(0.0, 1.0)


def _upright_gate(env, threshold=0.8):
    return (uprightness(env) > threshold).float()


def track_biped_lin_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track (vx, vy) in the heading frame; only counts while upright."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    err = torch.sum(torch.square(cmd - biped_lin_vel(env)), dim=1)
    return torch.exp(-err / std**2) * _upright_gate(env)


def track_biped_yaw_rate_exp(env: ManagerBasedRLEnv, command_name: str, std: float,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Track yaw rate about world z; only counts while upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    err = torch.square(cmd - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-err / std**2) * _upright_gate(env)


def ang_vel_world_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tipping rates (world roll/pitch rates) - frame independent of the body pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w[:, :2]), dim=1)


def contact_count(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Number of selected bodies in contact (used for 'front feet must stay off the ground')."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0]
    return (f > threshold).float().sum(dim=1)


def hind_feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg,
                       threshold: float) -> torch.Tensor:
    """Reward steps of the hind feet when a motion command is given (biped stepping)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    r = torch.sum((air - threshold).clamp(max=0.3) * first_contact, dim=1)
    moving = torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return r * moving * _upright_gate(env)


def hind_feet_stance_width(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target: float) -> torch.Tensor:
    """Keep the two hind feet ~`target` m apart horizontally (avoids crossing / too narrow stance)."""
    asset: Articulation = env.scene[asset_cfg.name]
    p = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    d = (p[:, 0] - p[:, 1]).norm(dim=1)
    return torch.square(d - target)


# ----------------------------------------------------------------------------- events
def reset_quad_or_biped(env: ManagerBasedRLEnv, env_ids: torch.Tensor, biped_prob: float,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                        joint_overrides: dict | None = None, mid_prob: float = 0.0,
                        mid_range: tuple[float, float] = (0.2, 0.9), quad_joint_noise: float = 0.05) -> torch.Tensor:
    """Reset each env either in the default quadruped stance or already standing on the hind legs.

    Starting part of the episodes upright makes balance learnable early; the rest must learn the
    stand-up transition from four legs. With ``mid_prob`` > 0, a fraction of the non-biped resets
    starts *between* the two poses (pitch 18-81 deg, joints interpolated): a state-distribution
    curriculum that exposes the policy to the stand-up transition. ``quad_joint_noise``: joint spread of the
    quadruped resets (+-, rad; the others keep 0.05).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = env.device
    u = torch.rand(n, device=dev)
    biped = u < biped_prob
    mid = (~biped) & (u < biped_prob + mid_prob)
    # f: 0 = quadruped pose, 1 = upright biped pose
    f = torch.where(biped, torch.ones(n, device=dev), torch.zeros(n, device=dev))
    f = torch.where(mid, sample_uniform(mid_range[0], mid_range[1], (n,), dev), f)

    root = asset.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, :2] += sample_uniform(-0.5, 0.5, (n, 2), dev)
    yaw = sample_uniform(-math.pi, math.pi, (n,), dev)
    zeros = torch.zeros(n, device=dev)
    q_yaw = quat_from_euler_xyz(zeros, zeros, yaw)
    pitch = -f * math.pi / 2 + sample_uniform(-0.1, 0.1, (n,), dev) * (f > 0)  # small tilt noise unless quad
    q_pitch = quat_from_euler_xyz(zeros, pitch, zeros)
    root[:, 3:7] = quat_mul(q_yaw, q_pitch)
    z_mid = 0.36 + 0.2 * torch.sin(-pitch).clamp(min=0.0)  # hind hips stay ~on the legs while pitching up
    root[:, 2] = torch.where(biped, torch.full((n,), UPRIGHT_BASE_HEIGHT, device=dev), root[:, 2])
    root[:, 2] = torch.where(mid, z_mid, root[:, 2])
    root[:, 7:] = 0.0
    asset.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

    jpos = asset.data.default_joint_pos[env_ids].clone()
    up = jpos.clone()
    for expr, val in {**UPRIGHT_JOINT_POS, **(joint_overrides or {})}.items():
        ids, _ = asset.find_joints(expr)
        up[:, ids] = val
    jpos = jpos + f[:, None] * (up - jpos)  # interpolate quad -> upright joint configuration
    spread = torch.where(f == 0, torch.full((n,), quad_joint_noise, device=dev), torch.full((n,), 0.05, device=dev))
    jpos += sample_uniform(-1.0, 1.0, jpos.shape, dev) * spread[:, None]
    jvel = torch.zeros_like(jpos)
    asset.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
    return root


# ============================================================================ carrying (stage 3)
# Where the held object should sit, in the robot base frame (upright biped: +x up, -z = belly side).
OBJ_TARGET_B = (0.11, 0.0, -0.25)  # matches the front-foot positions of HOLD_JOINT_POS (measured in sim)
# Front-foot positions of the hold pose in the base frame (scripts/diagnostics/diag_carry_init.py); y is squeezed a bit
# inside the object half-width so the feet press on the object.
HOLD_FEET_B = {"FL_foot": (0.11, 0.085, -0.25), "FR_foot": (0.11, -0.085, -0.25)}
# Front legs reaching forward around the object, slightly squeezed inwards.
HOLD_JOINT_POS = {
    "F[L,R]_thigh_joint": 1.2,
    "F[L,R]_calf_joint": -1.8,
    "FL_hip_joint": -0.15,
    "FR_hip_joint": 0.15,
}


def _obj_target_w(env, robot_cfg=SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    t = torch.tensor(OBJ_TARGET_B, device=env.device).expand(env.num_envs, 3)
    return robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, t)


def object_pos_b(env: ManagerBasedRLEnv, robot_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object")):
    """Object position relative to the robot base, in the base frame."""
    from isaaclab.utils.math import quat_apply_inverse
    robot: Articulation = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)


def object_rel_vel_b(env: ManagerBasedRLEnv, robot_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object")):
    from isaaclab.utils.math import quat_apply_inverse
    robot: Articulation = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)


def object_mass(env: ManagerBasedRLEnv, object_cfg=SceneEntityCfg("object")):
    obj = env.scene[object_cfg.name]
    return obj.root_physx_view.get_masses().to(env.device).reshape(env.num_envs, -1)[:, :1]


def object_hold_reward(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    obj = env.scene["object"]
    err = torch.sum(torch.square(obj.data.root_pos_w - _obj_target_w(env)), dim=1)
    return torch.exp(-err / std**2)


def object_rel_vel_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(object_rel_vel_b(env)), dim=1)


def object_grip(env: ManagerBasedRLEnv, sensor_names: list[str], threshold: float = 0.5) -> torch.Tensor:
    """Fraction of the front feet touching the object (one filtered contact sensor per foot)."""
    touching = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        f = sensor.data.force_matrix_w.norm(dim=-1).sum(dim=(1, 2))  # (N,)
        touching.append((f > threshold).float())
    return torch.stack(touching, dim=1).mean(dim=1)


def front_feet_hold_pose(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Dense shaping: front feet near their hold positions (base frame), so the arms come down to the object."""
    from isaaclab.utils.math import quat_apply_inverse
    robot: Articulation = env.scene["robot"]
    err = torch.zeros(env.num_envs, device=env.device)
    for name, target in HOLD_FEET_B.items():
        i = robot.find_bodies(name)[0][0]
        p_b = quat_apply_inverse(robot.data.root_quat_w, robot.data.body_pos_w[:, i] - robot.data.root_pos_w)
        err += torch.sum(torch.square(p_b - torch.tensor(target, device=env.device)), dim=1)
    return torch.exp(-err / std**2)


def feet_below(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, height: float) -> torch.Tensor:
    """Penalty for selected feet lower than `height` (front feet must stay up, even while holding)."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    return (height - z).clamp(min=0.0).sum(dim=1)


def object_dropped(env: ManagerBasedRLEnv, max_dist: float, min_height: float) -> torch.Tensor:
    obj = env.scene["object"]
    far = (obj.data.root_pos_w - _obj_target_w(env)).norm(dim=1) > max_dist
    low = (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) < min_height
    return far | low


def reset_biped_holding_object(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Robot upright with front legs around the object; object placed at the hold target."""
    root = reset_quad_or_biped(env, env_ids, biped_prob=1.0, joint_overrides=HOLD_JOINT_POS)
    obj = env.scene["object"]
    t = torch.tensor(OBJ_TARGET_B, device=env.device).expand(len(env_ids), 3)
    pose = torch.cat([root[:, :3] + quat_apply(root[:, 3:7], t), root[:, 3:7]], dim=1)
    obj.write_root_pose_to_sim(pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def object_gravity_assist(env: ManagerBasedRLEnv, env_ids: torch.Tensor, start_step: int, end_step: int) -> None:
    """Curriculum: cancel a fraction of the object's weight with an upward force.

    assist = 1 (weightless) until ``start_step`` env steps, then decays linearly to 0 at ``end_step``.
    Lets the policy first learn *where* to put the front feet before having to carry the full load.
    """
    c = env.common_step_counter
    assist = 1.0 if c < start_step else max(0.0, 1.0 - (c - start_step) / max(1, end_step - start_step))
    obj = env.scene["object"]
    m = obj.root_physx_view.get_masses().to(env.device).reshape(env.num_envs, -1)[:, :1]  # (N, 1)
    forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    forces[:, 0, 2] = assist * 9.81 * m[:, 0]
    obj.set_external_force_and_torque(forces, torch.zeros_like(forces), is_global=True)
    env.extras.setdefault("log", {})["Curriculum/object_gravity_assist"] = assist


def object_spring_assist(env: ManagerBasedRLEnv, env_ids: torch.Tensor, start_step: int, end_step: int,
                         kp: float = 100.0, kd: float = 20.0) -> None:
    """Curriculum 'training wheels': a mass-scaled spring-damper pulls the object to the hold target and
    cancels gravity. Unlike a weightless object, touching it does not push it away, so the policy can
    safely explore squeezing. Strength decays 1 -> 0 between ``start_step`` and ``end_step`` env steps.
    (carry_v2 with a plain weightless object learned to *avoid* touching it.)
    """
    c = env.common_step_counter
    a = 1.0 if c < start_step else max(0.0, 1.0 - (c - start_step) / max(1, end_step - start_step))
    obj = env.scene["object"]
    robot: Articulation = env.scene["robot"]
    m = obj.root_physx_view.get_masses().to(env.device).reshape(env.num_envs, -1)[:, :1]  # (N, 1)
    dp = _obj_target_w(env) - obj.data.root_pos_w
    dv = robot.data.root_lin_vel_w - obj.data.root_lin_vel_w
    acc = kp * dp + kd * dv
    acc[:, 2] += 9.81
    forces = (a * m * acc).unsqueeze(1)
    obj.set_external_force_and_torque(forces, torch.zeros_like(forces), is_global=True)
    env.extras.setdefault("log", {})["Curriculum/object_assist"] = a


def object_squeeze(env: ManagerBasedRLEnv, sensor_names: list[str], force_cap: float) -> torch.Tensor:
    """Reward squeezing the object from BOTH sides: min(left, right) contact force, normalized and capped."""
    forces = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        forces.append(sensor.data.force_matrix_w.norm(dim=-1).sum(dim=(1, 2)))
    return (torch.stack(forces, dim=1).min(dim=1)[0] / force_cap).clamp(max=1.0)


class AdaptiveObjectAssist:
    """Performance-driven version of :func:`object_spring_assist`.

    The assist level starts at ``init``; every ``window`` env steps it is lowered by ``step`` if the mean
    hold quality (``object_hold_reward``, 1 = object exactly at the target) over the window is above
    ``lower_if_above`` and raised if it is below ``raise_if_below``. (carry_v3's fixed schedule removed
    the assist faster than the policy learned to squeeze; a first version keyed on termination counts
    read the termination buffers at the wrong time and only ever raised the assist.)
    """

    level: float | None = None
    acc = 0.0
    n = 0
    last_update = 0


def adaptive_object_assist(env: ManagerBasedRLEnv, env_ids: torch.Tensor, init: float, step: float, window: int,
                           lower_if_above: float, raise_if_below: float, hold_std: float = 0.08,
                           kp: float = 100.0, kd: float = 20.0) -> None:
    st = AdaptiveObjectAssist
    if st.level is None:
        st.level = init
    st.acc += float(object_hold_reward(env, hold_std).mean())
    st.n += 1
    c = env.common_step_counter
    if c - st.last_update >= window and st.n > 0:
        q = st.acc / st.n
        if q > lower_if_above:
            st.level = max(0.0, st.level - step)
        elif q < raise_if_below:
            st.level = min(1.0, st.level + step)
        env.extras.setdefault("log", {})["Curriculum/hold_quality"] = q
        st.acc, st.n, st.last_update = 0.0, 0, c
    a = st.level
    obj = env.scene["object"]
    robot: Articulation = env.scene["robot"]
    m = obj.root_physx_view.get_masses().to(env.device).reshape(env.num_envs, -1)[:, :1]
    dp = _obj_target_w(env) - obj.data.root_pos_w
    dv = robot.data.root_lin_vel_w - obj.data.root_lin_vel_w
    acc = kp * dp + kd * dv
    acc[:, 2] += 9.81
    forces = (a * m * acc).unsqueeze(1)
    obj.set_external_force_and_torque(forces, torch.zeros_like(forces), is_global=True)
    env.extras.setdefault("log", {})["Curriculum/object_assist"] = a


# ============================================================================ unknown loads on the forearms
FOREARMS = ["FL_calf", "FR_calf"]


class ForearmLoad:
    """Per-env state of the simulated payload resting on the front forearms (world-frame downward forces)."""

    force: torch.Tensor | None = None  # (N, 2) current downward force [N] on FL_calf / FR_calf
    target: torch.Tensor | None = None  # (N, 2) force the load is ramping towards
    rate: torch.Tensor | None = None  # (N, 1) ramp speed [N/s] (inf = sudden)
    timer: torch.Tensor | None = None  # (N,) seconds until the load changes


def _forearm_ids(robot):
    return robot.find_bodies(FOREARMS, preserve_order=True)[0]


def _sample_load(n, dev, mass_range, p_none, asym):
    m = sample_uniform(mass_range[0], mass_range[1], (n,), dev)
    m = torch.where(torch.rand(n, device=dev) < p_none, torch.zeros_like(m), m)
    split = 0.5 + sample_uniform(-asym, asym, (n,), dev) * 0.5  # share on the left forearm
    w = m * 9.81
    return torch.stack([w * split, w * (1 - split)], dim=1)


def forearm_load(env: ManagerBasedRLEnv, env_ids: torch.Tensor, mass_range: tuple[float, float],
                 hold_range_s: tuple[float, float], p_none: float, asym: float,
                 ramp_range_s: tuple[float, float] = (0.0, 1.5), p_sudden: float = 0.3,
                 fixed_mass: float | None = None, fixed_on_time_s: float = 0.0, fixed_ramp_s: float = 0.0) -> None:
    """Unknown payload on the forearms that is placed / removed / changed every few seconds.

    Applied every policy step as a world-frame downward force on the two front calves (like a box resting on
    the forearms). Each change ramps in over a random time in ``ramp_range_s`` (a person handing the box over
    slowly) or, with probability ``p_sudden``, instantly (dropped onto the arms).
    For evaluation, ``fixed_mass`` ramps a constant load on from ``fixed_on_time_s`` over ``fixed_ramp_s``.
    """
    st = ForearmLoad
    robot: Articulation = env.scene["robot"]
    n, dev = env.num_envs, env.device
    if st.force is None or st.force.shape[0] != n:
        st.force = torch.zeros(n, 2, device=dev)
        st.target = torch.zeros(n, 2, device=dev)
        st.rate = torch.full((n, 1), float("inf"), device=dev)
        st.timer = sample_uniform(0.5, 3.0, (n,), dev)
    # episodes that just restarted: start unloaded, load arrives a bit later (= placed while standing)
    fresh = env.episode_length_buf <= 1
    st.force[fresh] = 0.0
    st.target[fresh] = 0.0
    st.timer[fresh] = sample_uniform(0.5, 3.0, (int(fresh.sum()),), dev)
    if fixed_mass is not None:
        t = env.episode_length_buf.float() * env.step_dt
        frac = ((t - fixed_on_time_s) / max(fixed_ramp_s, 1e-6)).clamp(0.0, 1.0)
        st.force = (fixed_mass * 9.81 / 2) * frac[:, None].expand(n, 2).clone()
    else:
        st.timer -= env.step_dt
        due = st.timer <= 0
        k = int(due.sum())
        if k:
            st.target[due] = _sample_load(k, dev, mass_range, p_none, asym)
            ramp = sample_uniform(ramp_range_s[0], ramp_range_s[1], (k,), dev)
            ramp = torch.where(torch.rand(k, device=dev) < p_sudden, torch.zeros_like(ramp), ramp)
            delta = (st.target[due] - st.force[due]).abs().max(dim=1, keepdim=True)[0]
            st.rate[due] = torch.where(ramp[:, None] > 0, delta / ramp[:, None].clamp(min=1e-3),
                                       torch.full_like(delta, float("inf")))
            st.timer[due] = sample_uniform(hold_range_s[0], hold_range_s[1], (k,), dev)
        step = st.rate * env.step_dt
        st.force = st.force + (st.target - st.force).clamp(min=-step, max=step)
    ids = _forearm_ids(robot)
    forces = torch.zeros(n, len(ids), 3, device=dev)
    forces[..., 2] = -st.force
    robot.set_external_force_and_torque(forces, torch.zeros_like(forces), body_ids=ids, is_global=True)
    env.extras.setdefault("log", {})["Metrics/forearm_load_kg"] = float(st.force.sum(dim=1).mean() / 9.81)


def forearm_load_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged (critic-only) current forearm loads, scaled to ~[0, 1]."""
    if ForearmLoad.force is None or ForearmLoad.force.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, 2, device=env.device)
    return ForearmLoad.force / 10.0


def loaded_com_over_hind_feet(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Support-polygon term (cf. arXiv 2507.20382): horizontal distance between the combined CoM
    (robot + forearm payload) and the midpoint of the hind feet; exp kernel, only while upright."""
    robot: Articulation = env.scene["robot"]
    m = robot.root_physx_view.get_masses().to(env.device)  # (N, B)
    com = robot.data.body_com_pos_w  # (N, B, 3)
    tot_m = m.sum(dim=1, keepdim=True)
    xy = (m[..., None] * com[..., :2]).sum(dim=1)
    if ForearmLoad.force is not None and ForearmLoad.force.shape[0] == env.num_envs:
        lm = ForearmLoad.force / 9.81  # (N, 2) kg
        xy = xy + (lm[..., None] * robot.data.body_com_pos_w[:, _forearm_ids(robot), :2]).sum(dim=1)
        tot_m = tot_m + lm.sum(dim=1, keepdim=True)
    xy = xy / tot_m
    feet = robot.data.body_pos_w[:, robot.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)[0], :2].mean(dim=1)
    d2 = torch.sum(torch.square(xy - feet), dim=1)
    return torch.exp(-d2 / std**2) * _upright_gate(env)


# ============================================================================ gait (alternating hind feet)
def biped_feet_gait(env: ManagerBasedRLEnv, period: float, offset: list[float], sensor_cfg: SceneEntityCfg,
                    threshold: float, command_name: str) -> torch.Tensor:
    """unitree_rl_lab's feet_gait (contact must match a clock-driven stance schedule; offsets [0, 0.5] =
    left/right alternate), only while upright and moving."""
    from unitree_rl_lab.tasks.locomotion.mdp import feet_gait
    return feet_gait(env, period, offset, sensor_cfg, threshold, command_name) * _upright_gate(env)


def biped_swing_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float,
                       tanh_mult: float, command_name: str) -> torch.Tensor:
    """Swinging hind feet should clear ``target_height`` (world z); only while upright and moving."""
    from unitree_rl_lab.tasks.locomotion.mdp import foot_clearance_reward
    moving = torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return foot_clearance_reward(env, asset_cfg, target_height, std, tanh_mult) * moving * _upright_gate(env)


def biped_both_feet_when_still(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Zero command: stand on both hind feet (no stepping on the spot)."""
    from unitree_rl_lab.tasks.locomotion.mdp import feet_contact_without_cmd
    return feet_contact_without_cmd(env, sensor_cfg, command_name) * _upright_gate(env)


def hind_joint_pose_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target: list[float]) -> torch.Tensor:
    """Squared deviation of the hind thigh/calf joints from a knees-bent biped pose (cf. go2_quad2hand
    ``rear_dof_pos_hand``); without it the policy locks the knees straight. Only while upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    tgt = torch.tensor(target, device=env.device)
    return torch.sum(torch.square(q - tgt), dim=1) * _upright_gate(env)


def height_target_exp(env: ManagerBasedRLEnv, target: float, std: float,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp kernel around a target base height (unlike ``height_reward``, standing taller is not better)."""
    z = base_height(env, asset_cfg)[:, 0]
    return torch.exp(-torch.square(z - target) / std**2)


def biped_drift_when_still(env: ManagerBasedRLEnv, command_name: str,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Zero command: penalize horizontal base velocity and yaw rate (the v1 policy drifted 1.5 m / 10 s)."""
    asset: Articulation = env.scene[asset_cfg.name]
    still = torch.norm(env.command_manager.get_command(command_name), dim=1) <= 0.1
    v = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1) + 0.5 * torch.square(asset.data.root_ang_vel_w[:, 2])
    return v * still * _upright_gate(env)


def hind_swing_knee_flex(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, period: float, offset: list[float],
                         stance_frac: float, straight_calf: float, flex_target: float,
                         command_name: str) -> torch.Tensor:
    """Bend the knee *while the foot swings* (clearance by knee flexion, not a crouched stance).
    Same clock as ``feet_gait``: a leg swings when its phase >= ``stance_frac``. Per leg,
    sin(pi * swing progress) * clamp(knee bend from straight / flex_target, 0, 1): peaks mid-swing,
    ~0 at lift-off / touch-down, so the leg can still land extended. asset_cfg: [RL_calf, RR_calf]."""
    asset: Articulation = env.scene[asset_cfg.name]
    g = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.stack([(g + o) % 1.0 for o in offset], dim=1)
    s = ((phase - stance_frac) / (1.0 - stance_frac)).clamp(0.0, 1.0)
    shape = torch.sin(torch.pi * s) * (phase >= stance_frac)
    bend = (straight_calf - asset.data.joint_pos[:, asset_cfg.joint_ids]) / flex_target  # calf angle goes negative to bend
    r = torch.sum(shape * bend.clamp(0.0, 1.0), dim=1)
    moving = torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return r * moving * _upright_gate(env)


def lin_vel_z_world_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vertical base velocity in the world frame (Isaac Lab's lin_vel_z_l2 uses body z, which is horizontal
    when the robot stands on its hind legs). Only while upright, so standing up is not penalized."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2]) * _upright_gate(env)


# ----------------------------------------------------------------------------- heading tracking (v6)
class HeadingTarget:
    """Per-env desired biped heading [rad], integrated from the yaw-rate command (joystick-like)."""

    psi_des: torch.Tensor | None = None


def _biped_yaw(env) -> torch.Tensor:
    h = biped_heading_xy(env)
    return torch.atan2(h[:, 1], h[:, 0])


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def update_heading_target(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str, leash: float) -> None:
    """Interval event (every step): psi_des += wz_cmd * dt. Follows the actual heading while not upright or
    right after a reset, and is kept within ``leash`` rad of it so a lagging robot is not lost for good.
    Tracking the integrated heading instead of the instantaneous yaw rate is robust to the per-step torso
    twist of a biped (v5: 0.7-1.1 rad/s wobble hid the mean turn rate; cf. Siekmann et al. 2021, Cassie)."""
    st = HeadingTarget
    psi = _biped_yaw(env)
    if st.psi_des is None or st.psi_des.shape[0] != env.num_envs:
        st.psi_des = psi.clone()
    wz = env.command_manager.get_command(command_name)[:, 2]
    st.psi_des = st.psi_des + wz * env.step_dt
    follow = (env.episode_length_buf <= 1) | (uprightness(env) <= 0.8)
    e = _wrap(st.psi_des - psi).clamp(-leash, leash)
    st.psi_des = torch.where(follow, psi, psi + e)


def heading_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: wrap(psi_des - psi), shape (N, 1)."""
    if HeadingTarget.psi_des is None or HeadingTarget.psi_des.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return _wrap(HeadingTarget.psi_des - _biped_yaw(env))[:, None]


def track_heading_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    return torch.exp(-torch.square(heading_error(env)[:, 0]) / std**2) * _upright_gate(env)


def lin_vel_cmd_levels_thr(env: ManagerBasedRLEnv, env_ids, reward_term_name: str, threshold: float) -> torch.Tensor:
    """unitree_rl_lab's lin_vel_cmd_levels with a configurable threshold (it hard-codes 0.8 * weight).
    With a tighter tracking kernel the same tracking quality scores lower: Go2's 0.8 at std^2 0.25 is
    err^2 = 0.056, which scores exp(-0.056 / 0.1) = 0.57 at std^2 0.1 -> use ~0.6 there."""
    term = env.command_manager.get_term("base_velocity")
    ranges, lim = term.cfg.ranges, term.cfg.limit_ranges
    w = env.reward_manager.get_term_cfg(reward_term_name).weight
    rew = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0 and rew > w * threshold:
        d = torch.tensor([-0.1, 0.1], device=env.device)
        ranges.lin_vel_x = torch.clamp(torch.tensor(ranges.lin_vel_x, device=env.device) + d,
                                       lim.lin_vel_x[0], lim.lin_vel_x[1]).tolist()
        ranges.lin_vel_y = torch.clamp(torch.tensor(ranges.lin_vel_y, device=env.device) + d,
                                       lim.lin_vel_y[0], lim.lin_vel_y[1]).tolist()
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def track_biped_lin_vel_y_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Lateral-only tracking (heading frame), its own kernel: the shared xy kernel (std^2 0.1) scored a 0.08 m/s
    response to a 0.2 m/s side command at 82 % (Load-V5), too flat to pay for the extra joint effort."""
    cmd = env.command_manager.get_command(command_name)[:, 1]
    return torch.exp(-torch.square(cmd - biped_lin_vel(env)[:, 1]) / std**2) * _upright_gate(env)


# ----------------------------------------------------------------------------- real box on the forearms (Box task)
# Front-leg "tray" pose: forearms (calves) horizontal in front of the belly. Planar FK with thigh = calf = 0.213 m:
# knee at (fwd 0.213 cos 1.3, up -0.213 sin 1.3) from the shoulder, foot 0.213 m further forward at the same height
# (thigh + calf = 0). Arms ~0.28 m apart (hip + thigh offsets).
TRAY_JOINT_POS = {"F[L,R]_hip_joint": 0.0, "F[L,R]_thigh_joint": 1.3, "F[L,R]_calf_joint": -1.3}
BOX_H_MAX = 0.16  # tallest box; boxes are placed for this height (shorter ones drop up to 4 cm extra)
# box center this far in front of the base: belly at ~0.057 m (trunk half-thickness) + deepest box 0.25 / 2 + 1 cm,
# so the box does not start inside the trunk (forearms span 0.055-0.268 m in the tray pose)
BOX_PLACE_FWD = 0.19
FOREARM_RADIUS = 0.02


class BoxState:
    present: torch.Tensor | None = None  # (N,) this episode gets a box
    placed: torch.Tensor | None = None  # (N,) box has been handed over
    t_place: torch.Tensor | None = None  # (N,) s
    drop: torch.Tensor | None = None  # (N,) extra drop height [m]


def _heading_frame(env):
    h = biped_heading_xy(env)
    fwd = torch.cat([h, torch.zeros_like(h[:, :1])], 1)
    left = torch.stack([-h[:, 1], h[:, 0], torch.zeros_like(h[:, 0])], 1)
    return fwd, left


def forearm_center_w(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(center (N,3), top height (N,)) of the two forearm segments (knee -> foot), world frame."""
    robot: Articulation = env.scene["robot"]
    ids = robot.find_bodies(["FL_calf", "FL_foot", "FR_calf", "FR_foot"], preserve_order=True)[0]
    p = robot.data.body_pos_w[:, ids]
    return p.mean(dim=1), p[..., 2].max(dim=1)[0] + FOREARM_RADIUS


def box_handover(env: ManagerBasedRLEnv, env_ids: torch.Tensor, p_box: float, place_range_s: tuple[float, float],
                 p_gentle: float, drop_range: tuple[float, float], park_height: float = 3.0,
                 fixed_place_s: float | None = None, fixed_drop: float | None = None) -> None:
    """Interval event (every step). Each episode: with prob ``p_box`` a box is placed onto the forearms at a random
    time (a person handing it over): gently (resting on the arms) or dropped from up to ``drop_range[1]``.
    Until then the box is parked high above the env (re-written every step so it never falls)."""
    st = BoxState
    obj = env.scene["object"]
    n, dev = env.num_envs, env.device
    if st.present is None or st.present.shape[0] != n:
        st.present = torch.zeros(n, dtype=torch.bool, device=dev)
        st.placed = torch.zeros(n, dtype=torch.bool, device=dev)
        st.t_place = torch.zeros(n, device=dev)
        st.drop = torch.zeros(n, device=dev)
    fresh = env.episode_length_buf <= 1
    k = int(fresh.sum())
    if k:
        st.present[fresh] = torch.rand(k, device=dev) < p_box
        st.placed[fresh] = False
        st.t_place[fresh] = (sample_uniform(*place_range_s, (k,), dev) if fixed_place_s is None
                             else torch.full((k,), fixed_place_s, device=dev))
        d = sample_uniform(*drop_range, (k,), dev)
        d = torch.where(torch.rand(k, device=dev) < p_gentle, torch.zeros_like(d), d)
        st.drop[fresh] = d if fixed_drop is None else torch.full((k,), fixed_drop, device=dev)
    t = env.episode_length_buf.float() * env.step_dt
    due = st.present & ~st.placed & (t >= st.t_place) & (uprightness(env) > 0.8)
    parked = ~st.placed & ~due
    if parked.any():
        ids = parked.nonzero(as_tuple=True)[0]
        pose = torch.zeros(len(ids), 7, device=dev)
        pose[:, :3] = env.scene.env_origins[ids] + torch.tensor([0.0, 0.0, park_height], device=dev)
        pose[:, 3] = 1.0
        obj.write_root_pose_to_sim(pose, env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)
    if due.any():
        ids = due.nonzero(as_tuple=True)[0]
        c, top = forearm_center_w(env)
        robot: Articulation = env.scene["robot"]
        fwd, left = _heading_frame(env)
        base = robot.data.root_pos_w[ids]
        lat = ((c - robot.data.root_pos_w) * left).sum(1)[ids]
        pos = base + BOX_PLACE_FWD * fwd[ids] + lat[:, None] * left[ids]
        pos[:, 2] = top[ids] + BOX_H_MAX / 2 + st.drop[ids]
        yaw = _biped_yaw(env)[ids]
        z = torch.zeros_like(yaw)
        pose = torch.cat([pos, quat_from_euler_xyz(z, z, yaw)], dim=1)
        obj.write_root_pose_to_sim(pose, env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)
        st.placed[ids] = True


def _box_active(env) -> torch.Tensor:
    st = BoxState
    if st.placed is None or st.placed.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return st.placed


def box_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Critic obs: box position relative to the forearm center, heading frame (fwd, left, up); 0 without a box."""
    obj = env.scene["object"]
    c, _ = forearm_center_w(env)
    fwd, left = _heading_frame(env)
    d = obj.data.root_pos_w - c
    v = torch.stack([(d * fwd).sum(1), (d * left).sum(1), d[:, 2]], 1)
    return v * _box_active(env)[:, None]


def box_rel_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj = env.scene["object"]
    robot: Articulation = env.scene["robot"]
    fwd, left = _heading_frame(env)
    d = obj.data.root_lin_vel_w - robot.data.root_lin_vel_w
    v = torch.stack([(d * fwd).sum(1), (d * left).sum(1), d[:, 2]], 1)
    return v * _box_active(env)[:, None]


def box_mass_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return object_mass(env) * _box_active(env)[:, None]


def box_present_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _box_active(env).float()[:, None]


def box_on_forearms(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Box resting on the arms: horizontally over the forearm center and not below the forearm top."""
    obj = env.scene["object"]
    c, top = forearm_center_w(env)
    dh = torch.sum(torch.square((obj.data.root_pos_w - c)[:, :2]), dim=1)
    above = (obj.data.root_pos_w[:, 2] > top).float()
    return torch.exp(-dh / std**2) * above * _box_active(env)


def box_tilt(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 - cos(box tilt); only with a box."""
    obj = env.scene["object"]
    up = quat_apply(obj.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    return (1.0 - up[:, 2]) * _box_active(env)


def box_dropped(env: ManagerBasedRLEnv, max_horizontal: float, max_below: float) -> torch.Tensor:
    obj = env.scene["object"]
    c, top = forearm_center_w(env)
    far = (obj.data.root_pos_w - c)[:, :2].norm(dim=1) > max_horizontal
    low = obj.data.root_pos_w[:, 2] < top - max_below
    return (far | low) & _box_active(env)


def front_tray_pose_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target: list[float]) -> torch.Tensor:
    """G1-style joint deviation (L1) of the front legs, around the tray pose instead of the default pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(q - torch.tensor(target, device=env.device)), dim=1) * _upright_gate(env)


def com_over_feet_with_box(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """loaded_com_over_hind_feet, with the box (once on the arms) in the combined CoM."""
    robot: Articulation = env.scene["robot"]
    m = robot.root_physx_view.get_masses().to(env.device)
    xy = (m[..., None] * robot.data.body_com_pos_w[..., :2]).sum(dim=1)
    tot = m.sum(dim=1, keepdim=True)
    act = _box_active(env).float()[:, None]
    bm = object_mass(env) * act
    xy = xy + bm * env.scene["object"].data.root_pos_w[:, :2]
    xy = xy / (tot + bm)
    feet = robot.data.body_pos_w[:, robot.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)[0], :2].mean(dim=1)
    return torch.exp(-torch.sum(torch.square(xy - feet), dim=1) / std**2) * _upright_gate(env)


def body_touches_ground(env: ManagerBasedRLEnv, sensor_names: list[str], threshold: float = 1.0) -> torch.Tensor:
    """Fall termination for the Box task: base / head touching the GROUND (filtered sensors), so a box resting
    against the belly does not count as a fall (illegal_contact counts any contact)."""
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in sensor_names:
        f = env.scene.sensors[name].data.force_matrix_w  # (N, 1, 1, 3)
        hit |= f.norm(dim=-1).reshape(env.num_envs, -1).max(dim=1)[0] > threshold
    return hit
