"""RefWalk: hind-leg walking rebuilt from go2_quad2hand's biped mode (2026-10-20).

Reference: ~/go2_sim2real/refs/go2_quad2hand, go2_rl/legged_gym/envs/go2/go2_quad2hand_config.py + go2_quad2hand.py
(called "ref" below). The reference walks on the FRONT legs (handstand, pitch 84 deg, rear feet must stay up);
stance legs <-> free legs are swapped here.

Item-by-item (ref -> here):
  action scale 0.25, PD 30 / 0.75, decimation 4, dt 0.005 ............... same
  commands (biped) vx +-0.4, vy 0, wz +-0.5, |c|<=0.1 -> 0, resample 10 s same (task: carry in the commanded
                                                                          direction; speed is not a goal)
  gait: period 0.4 s, offsets [0, 0.5], stance < 0.6, always on, 0.5 ... same (hind feet)
  tracking: sigma 0.25, lin 1.5 (vx only), yaw rate 1.0 ................ lin tracks (vx, vy) with vy = 0 so it also
                                                                          penalizes sideways drift; both gated on
                                                                          upright (else standing on 4 legs pays)
  base height exp(-(h-0.5)^2/0.25) 1.0, orientation 84 deg exp(/0.25) 1.0 same (lean +6 deg towards the belly)
  foot clearance 0.05 m (+0.022 offset), sigma 0.01, 0.2 ............... same (hind feet)
  contact_no_vel -0.1, hip_close_to_default -1.0 ........................ same
  free-leg contact -2.0 (front feet), free-leg dof pos -0.1 ............. same weights; dof target = tray pose
                                                                          (thigh 1.3, calf -1.6) instead of the quad
                                                                          default: the Hold stage receives objects there
  collision thigh/calf -1.5 (penalty only, no termination) ............. same
  torques -1e-4, dof_vel -1e-4, dof_acc -1e-7, action_rate -1e-3 ......... same
  termination -200; terminate on base / head / hip contact ............. same
  obs: ang_vel*0.25, gravity, cmd*(2,2,0.25), q, dq*0.05, a, clock; noise same; history 4 (+ current) same
  velocity estimator (lin vel, 3) ........................................ not used (rsl_rl 3.0.1 has none); the
                                                                          critic sees the true base velocity
  privileged: lin vel, gains, joint friction / armature ................ same subset (no material friction / mass /
                                                                          CoM / torque scale / motor offset)
  DR: friction 0.2-1.25, base mass +-1 kg, CoM +-1 cm, push 0.2 m/s + 0.4 rad/s every 8 s,
      joint friction 0.005-0.05, armature 0.01-0.05, Kp/Kd x0.9-1.1 per episode .... same
  DR joint damping 0.1-0.5 ............................................... not used: in Isaac Gym effort-drive mode
                                                                          the DOF damping property has (most likely)
                                                                          no effect; Kd x0.9-1.1 covers it
  init: quad stance + pitch/roll +-0.2 ................................... reset_quad_or_biped (50 % upright, 30 %
                                                                          between quad and upright), as v1-v6b: the
                                                                          hind-leg stand-up lifts the CoM much more
                                                                          than the handstand
  episode 20 s; PPO (512-256-128, lr 1e-3 adaptive, entropy 0.01, 24 steps) same; + L/R mirror augmentation
                                                                          (v1-v4 drifted asymmetric without it)
  quadruped mode (one policy, two gaits) ................................. not used here: added later in switch_env_cfg.py
"""
import math
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from . import mdp as bmdp
from . import refwalk_mdp as rmdp
from .biped_env_cfg import BipedSceneCfg, BipedStandEnvCfg

STEP_PERIOD = 0.4  # ref period_table[hand]
HISTORY = 5  # ref: current + history_len 4
FRONT_LEG_JOINTS = ["FL_hip_joint", "FR_hip_joint", "FL_thigh_joint", "FR_thigh_joint", "FL_calf_joint", "FR_calf_joint"]
TRAY_POSE = [0.0, 0.0, 1.3, 1.3, -1.6, -1.6]  # box_mdp.TRAY_TARGET (Hold stage receives objects in this pose)
HIND_FEET = ["RL_foot", "RR_foot"]
ALL_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


KNEE_RATIO = 1.5  # ref URDF: calf effort 35.55 / velocity 20.07 vs hip & thigh 23.7 / 30.1 (extra 1.5:1 knee gear)


def refwalk_actuators(robot, knee_ratio: float | None):
    """Kp 30 / Kd 0.75 (ref control.stiffness / damping). With knee_ratio, the calf joints get their own torque-speed
    curve: unitree_rl_lab's Go2HV curve (used for ALL joints there) with torques x ratio and speeds / ratio.
    RefWalk v1 had 23.4 Nm knees: every evaluation sat at that limit, and in MuJoCo (45.43 Nm knees, as the real
    robot) the policy over-extended and fell backwards within 3 s (no fall when MuJoCo was clipped to 23.5 Nm)."""
    act = robot.actuators["GO2HV"]
    act.stiffness, act.damping = 30.0, 0.75
    if knee_ratio is None:
        return
    act.joint_names_expr = [".*_hip_joint", ".*_thigh_joint"]
    knee = act.replace(joint_names_expr=[".*_calf_joint"], X1=act.X1 / knee_ratio, X2=act.X2 / knee_ratio,
                       Y1=act.Y1 * knee_ratio, Y2=act.Y2 * knee_ratio)
    robot.actuators["GO2HV_KNEE"] = knee


@configclass
class RefWalkSceneCfg(BipedSceneCfg):
    # (no extra fields here: InteractiveSceneCfg treats every field as a scene entity)
    def __post_init__(self):
        super().__post_init__()
        refwalk_actuators(self.robot, None)  # v1: all joints on the hip/thigh curve


@configclass
class RefWalkV2SceneCfg(BipedSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        refwalk_actuators(self.robot, KNEE_RATIO)


@configclass
class CommandsCfg:
    base_velocity = rmdp.DeadzoneVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=rmdp.DeadzoneVelocityCommandCfg.Ranges(lin_vel_x=(-0.4, 0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5)),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


CMD_SCALE = (2.0, 2.0, 0.25)  # ref commands_scale = (lin_vel, lin_vel, ang_vel) obs scales


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, scale=CMD_SCALE, clip=(-100, 100),
                                    params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": STEP_PERIOD})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = HISTORY

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100))
        biped_lin_vel = ObsTerm(func=bmdp.biped_lin_vel_obs, scale=2.0, clip=(-100, 100))
        base_height = ObsTerm(func=bmdp.base_height, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, scale=CMD_SCALE, clip=(-100, 100),
                                    params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": STEP_PERIOD})
        actuator_kp_scale = ObsTerm(func=rmdp.actuator_kp_scale)
        actuator_kd_scale = ObsTerm(func=rmdp.actuator_kd_scale)
        joint_friction = ObsTerm(func=rmdp.joint_friction)
        joint_armature = ObsTerm(func=rmdp.joint_armature)

        def __post_init__(self):
            self.history_length = HISTORY

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.2, 1.25),
                "dynamic_friction_range": (0.2, 1.25), "restitution_range": (0.0, 0.0), "num_buckets": 64},
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-1.0, 1.0), "operation": "add"},
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}},
    )
    # PhysX joint friction coefficient (the same PhysX property legged_gym's DOF friction sets); Isaac Lab applies the
    # range also to the dynamic / viscous coefficients (<= 0.05, negligible next to Kd 0.75)
    joint_params = EventTerm(
        func=mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "friction_distribution_params": (0.005, 0.05),
                "armature_distribution_params": (0.01, 0.05), "operation": "abs"},
    )
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    reset_robot = EventTerm(func=bmdp.reset_quad_or_biped, mode="reset",
                            params={"biped_prob": 0.5, "mid_prob": 0.3, "mid_range": (0.0, 0.9)})
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(8.0, 8.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2),
                                   "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}},
    )


@configclass
class RewardsCfg:
    # -- task
    tracking_lin_vel = RewTerm(func=bmdp.track_biped_lin_vel_exp, weight=1.5,
                               params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    tracking_ang_vel = RewTerm(func=bmdp.track_biped_yaw_rate_exp, weight=1.0,
                               params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    # -- biped ("hand") mode shaping
    gait_hand = RewTerm(func=rmdp.gait_hand, weight=0.5,
                        params={"period": STEP_PERIOD, "offset": [0.0, 0.5], "stance_frac": 0.6,
                                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FEET, preserve_order=True)})
    base_height_hand = RewTerm(func=rmdp.base_height_hand, weight=1.0, params={"target": 0.5, "sigma": 0.25})
    body_orientation_hand = RewTerm(func=rmdp.body_orientation_hand, weight=1.0,
                                    params={"lean_target": math.radians(90.0 - 84.0), "sigma": 0.25})
    foot_clearance_hand = RewTerm(func=rmdp.foot_clearance_hand, weight=0.2,
                                  params={"target": 0.05, "sigma": 0.01,
                                          "asset_cfg": SceneEntityCfg("robot", body_names=HIND_FEET, preserve_order=True)})
    front_contact_hand = RewTerm(func=bmdp.contact_count, weight=-2.0,
                                 params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot"])})
    front_dof_pos_hand = RewTerm(func=rmdp.joint_dev_l2, weight=-0.1,
                                 params={"asset_cfg": SceneEntityCfg("robot", joint_names=FRONT_LEG_JOINTS, preserve_order=True),
                                         "target": TRAY_POSE})
    contact_no_vel = RewTerm(func=rmdp.contact_no_vel, weight=-0.1,
                             params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_FEET, preserve_order=True),
                                     "asset_cfg": SceneEntityCfg("robot", body_names=ALL_FEET, preserve_order=True)})
    hip_close_to_default = RewTerm(func=rmdp.joint_dev_l2, weight=-1.0,
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")})
    # -- regularization
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    collision = RewTerm(func=mdp.undesired_contacts, weight=-1.5,
                        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"])})
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", ".*_hip"]), "threshold": 1.0},
    )


@configclass
class RefWalkEnvCfg(BipedStandEnvCfg):
    scene: RefWalkSceneCfg = RefWalkSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0  # legged_gym default


@configclass
class RefWalkV2EnvCfg(RefWalkEnvCfg):
    """v2: knee torque-speed curve as in the reference URDF (x1.5). Warm start from v1 (same observations / actions)."""
    scene: RefWalkV2SceneCfg = RefWalkV2SceneCfg(num_envs=4096, env_spacing=2.5)


# MuJoCo torsional friction 0.02 m ~ mu * r with mu ~0.6-1 -> r up to ~0.03; REFWALK_TORSION="a,b" overrides (tests)
TORSION_RANGE = tuple(float(v) for v in os.environ.get("REFWALK_TORSION", "0.0,0.04").split(","))


@configclass
class RefWalkV3EnvCfg(RefWalkV2EnvCfg):
    """v3: v2 + per-env torsional friction on the feet (MuJoCo condim 4 halved v2's turning)."""
    scene: RefWalkV2SceneCfg = RefWalkV2SceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        # 0-0.03 m (0.03 reproduced MuJoCo's condim-4 turn rate); REFWALK_TORSION overrides for tests
        rng = TORSION_RANGE if "REFWALK_TORSION" in os.environ else (0.0, 0.03)
        self.events.foot_torsion = EventTerm(func=rmdp.randomize_foot_torsional_friction, mode="prestartup",
                                             params={"radius_range": rng})


@configclass
class RefWalkV3PlayEnvCfg(RefWalkV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class RefWalkV2PlayEnvCfg(RefWalkV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class RefWalkPlayEnvCfg(RefWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


# waiting ("ready") pose of the front legs, chosen with scripts/diagnostics/diag_ready_pose.py (fixed base, 2026-10-22): hips 0.45 rad
# inwards (top view: forearm tips 4 cm apart, elbows 19 cm), thigh 1.0 / calf -1.6 (side view: V, forearms 37 deg up).
# It kept all 8 test shapes (gentle and 10 cm drops, placement noise, 4 repeats; tray pose: small ones 20 %) and
# tolerated 15 m/s^2 backwards (tray 9) / 13-14 sideways (tray 11-12).
READY_POSE = [-0.45, 0.45, 1.0, 1.0, -1.6, -1.6]  # FRONT_LEG_JOINTS order; FL hip negative = inwards
READY_JOINTS = {"FL_hip_joint": -0.45, "FR_hip_joint": 0.45, "F[L,R]_thigh_joint": 1.0, "F[L,R]_calf_joint": -1.6}


@configclass
class CurriculumV4Cfg:
    cmd_ranges = CurrTerm(func=rmdp.cmd_range_curriculum,
                          params={"lin_term": "tracking_lin_vel", "ang_term": "tracking_ang_vel",
                                  "lin_threshold": 0.8, "ang_threshold": 0.7})


@configclass
class RefWalkV4EnvCfg(RefWalkV3EnvCfg):
    """v4 (rebuild, 2026-10-22): v3 (ref-aligned, knee x1.5, torsion 0-0.03 m) + new waiting pose + lateral walking and
    command ranges grown by a curriculum (user: as fast as possible without failing too often).
    * commands: start at the ref biped range (vx +-0.4, vy 0, wz +-0.5); grow by 0.1 per episode length while tracking
      > 0.8 x weight (lin, Go2's lin_vel_cmd_levels threshold) / > 0.7 x weight (yaw: v2 scored 0.83 while tracking
      100 % in evaluation); limits vx -0.8..1.2, vy +-0.5, wz +-1.5 (motor / airframe safety, agreed 10-21);
      half of the resamples stay in the ref range.
    * front legs: ref's free-leg dof prior (-0.1) now targets READY_POSE; upright resets start there.
    * hip_close_to_default (ref -1.0) on the hind hips only: on the front hips it penalised the inward "ha" pose.
    * from scratch, up to 8000 iterations (ref 3000 + stand-up + range growth), evaluated every 1500."""

    def __post_init__(self):
        super().__post_init__()
        ref = rmdp.DeadzoneVelocityCommandCfg.Ranges(lin_vel_x=(-0.4, 0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5))
        c = self.commands.base_velocity
        c.ranges = rmdp.DeadzoneVelocityCommandCfg.Ranges(lin_vel_x=(-0.4, 0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5))
        c.base_ranges = ref
        c.p_base = 0.5
        c.limit_ranges = rmdp.DeadzoneVelocityCommandCfg.Ranges(lin_vel_x=(-0.8, 1.2), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.5, 1.5))
        self.curriculum = CurriculumV4Cfg()
        # hind-leg motor load (user 2026-10-22: the hind motors overheat in long biped sessions): (tau / peak)^2,
        # weight grown adaptively 0 -> 2 while tracking and falls stay fine (refwalk_mdp.load_penalty_curriculum)
        self.rewards.hind_load = RewTerm(func=rmdp.hind_torque_load, weight=0.0)
        # 0.05 per episode length up to 1.0: the first version (0.1 up to 2.0) reached 1.8 within one 1500-iteration
        # chunk and found the locked-knee exploit on the way
        self.curriculum.hind_load = CurrTerm(func=rmdp.load_penalty_curriculum,
                                             params={"term_name": "hind_load", "lin_term": "tracking_lin_vel",
                                                     "step": 0.05, "w_max": 1.0})
        # keep the hind joints out of the last 0.15 rad before their hard limits (weight: unitree_rl_lab Go2
        # dof_pos_limits -10); see rmdp.joint_limit_margin
        self.rewards.hind_limit_margin = RewTerm(
            func=rmdp.joint_limit_margin, weight=-10.0,
            params={"margin": 0.15, "asset_cfg": SceneEntityCfg("robot", joint_names=["R[L,R]_hip_joint", "R[L,R]_thigh_joint", "R[L,R]_calf_joint"])})
        # lateral-only tracking (Load-V7: weight 1.0, std^2 0.02): under the motor-load penalty the shared xy kernel
        # (std^2 0.25) let lateral walking collapse (0.07 m/s for a 0.3 m/s command, RefWalk-V4 chunk 2) at a small cost
        self.rewards.tracking_lin_vel_y = RewTerm(func=bmdp.track_biped_lin_vel_y_exp, weight=1.0,
                                                  params={"command_name": "base_velocity", "std": 0.02 ** 0.5})
        self.rewards.front_dof_pos_hand.params["target"] = READY_POSE
        self.rewards.hip_close_to_default.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["RL_hip_joint", "RR_hip_joint"])
        self.events.reset_robot.params["joint_overrides"] = READY_JOINTS


@configclass
class RefWalkV4PlayEnvCfg(RefWalkV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.p_base = 0.0


# HoldBall waiting pose (2026-10-24, scripts/diagnostics/diag_ready_pose.py, logs/pose_v4_h*_drop*.json, fixed base, 2 kg): "ha"
# opened (hips 0.15 in) and upper arms forward (thigh 0.5); forearm tips 22 cm apart (READY_POSE: 4 cm), 32 cm in
# front of the base. Spheres 0.30 / 0.36 / 0.44 m kept 1.00 / 1.00 / 0.88 placed gently and 0.62 / 0.75 / 0.38 from
# 10 cm (READY_POSE: 0.38 / 0 / 0 and 0 / 0 / 0); the best sphere sum of the 48 poses, tied with 0.15 / 0.5 / -1.3 and
# 0.15 / 0.3 / -1.3, but the best on the 0.44 m sphere placed gently. Boxes narrower than the tip gap fall through
# (b5 / b6 / b7: 0 / 0 / 0.2). Close to the ref's free-leg default (hip 0, thigh 0.8, calf -1.5).
BALL_POSE = [-0.15, 0.15, 0.5, 0.5, -1.6, -1.6]  # FRONT_LEG_JOINTS order
BALL_JOINTS = {"FL_hip_joint": -0.15, "FR_hip_joint": 0.15, "F[L,R]_thigh_joint": 0.5, "F[L,R]_calf_joint": -1.6}


@configclass
class RefWalkBallEnvCfg(RefWalkV4EnvCfg):
    """Walking base for HoldBall (2026-10-24): RefWalk-V4 with the front legs held in BALL_POSE instead of READY_POSE.

    Item by item (ref = go2_quad2hand biped mode / RefWalk-V4 / here):
      free-leg dof prior -0.1 ...... ref: default pose (0, 0.8, -1.5) / READY_POSE / BALL_POSE (closer to the ref)
      upright reset front legs ..... ref: default / READY_POSE / BALL_POSE
      everything else (commands + range curriculum, lateral term, hind load penalty + limit margin, torsion 0-0.03,
      lin kernel, PPO) ............. as RefWalk-V4
    Warm start from RefWalk-V4 chunk 5 (model_7495, the HoldV4 base) + 1500 iterations: the same adaptation length
    as refwalk_v2 (knee curve) and refwalk_v3 (torsion), one RefWalk-V4 chunk."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.front_dof_pos_hand.params["target"] = BALL_POSE
        self.events.reset_robot.params["joint_overrides"] = BALL_JOINTS


@configclass
class RefWalkBallPlayEnvCfg(RefWalkBallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.p_base = 0.0
