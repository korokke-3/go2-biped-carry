"""Go2 bipedal (hind-leg) standing / walking task, built on unitree_rl_lab's Go2 velocity task."""
import math

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2 import velocity_env_cfg as base

from . import mdp as bmdp


@configclass
class BipedSceneCfg(base.RobotSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        # Flat indoor floor; no height scanner needed.
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0,
            ),
        )
        self.height_scanner = None


@configclass
class CommandsCfg:
    # (vx, vy, wz) in the biped heading frame (see mdp.py). Stage 1 (stand): all zero.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        biped_lin_vel = ObsTerm(func=bmdp.biped_lin_vel_obs, clip=(-100, 100))
        base_height = ObsTerm(func=bmdp.base_height, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-0.5, 1.0), "operation": "add"},
    )
    reset_robot = EventTerm(func=bmdp.reset_quad_or_biped, mode="reset",
                            params={"biped_prob": 0.3, "mid_prob": 0.4, "mid_range": (0.0, 0.9)})
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    # -- posture (stand on the hind legs)
    upright = RewTerm(func=bmdp.upright_reward, weight=2.0)
    height = RewTerm(func=bmdp.height_reward, weight=1.0, params={"low": 0.3, "high": 0.5})
    front_feet_off_ground = RewTerm(
        func=bmdp.contact_count, weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[L,R]_foot")},
    )
    tip_rate = RewTerm(func=bmdp.ang_vel_world_xy_l2, weight=-0.05)
    hind_stance = RewTerm(func=bmdp.hind_feet_stance_width, weight=-2.0,
                          params={"asset_cfg": SceneEntityCfg("robot", body_names="R[L,R]_foot"), "target": 0.25})
    # -- task (stage 2+; zero commands in stage 1 => rewards standing still)
    track_lin_vel = RewTerm(func=bmdp.track_biped_lin_vel_exp, weight=1.5,
                            params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_yaw_rate = RewTerm(func=bmdp.track_biped_yaw_rate_exp, weight=0.75,
                             params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    hind_air_time = RewTerm(func=bmdp.hind_feet_air_time, weight=0.5,
                            params={"command_name": "base_velocity", "threshold": 0.3,
                                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[L,R]_foot")})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.1,
                         params={"asset_cfg": SceneEntityCfg("robot", body_names="R[L,R]_foot"),
                                 "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[L,R]_foot")})
    # -- hardware friendliness (motor load / smoothness)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    # -- safety
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"])},
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_.*"]), "threshold": 1.0},
    )


@configclass
class BipedStandEnvCfg(base.RobotEnvCfg):
    scene: BipedSceneCfg = BipedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        # (not calling base.RobotEnvCfg.__post_init__: it touches the removed height scanner / terrain curriculum)
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BipedStandPlayEnvCfg(BipedStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class BipedWalkEnvCfg(BipedStandEnvCfg):
    """Stage 2: follow (vx, vy, wz) commands in the biped heading frame (joystick-like)."""

    def __post_init__(self):
        super().__post_init__()
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.2
        cmd.resampling_time_range = (5.0, 8.0)
        cmd.ranges.lin_vel_x = (-0.3, 0.5)
        cmd.ranges.lin_vel_y = (-0.2, 0.2)
        cmd.ranges.ang_vel_z = (-0.8, 0.8)
        self.events.reset_robot.params.update({"biped_prob": 0.5, "mid_prob": 0.3, "mid_range": (0.0, 0.9)})


@configclass
class BipedWalkPlayEnvCfg(BipedWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


GAIT_PERIOD = 0.6  # s, one left + one right step


@configclass
class BipedGaitWalkEnvCfg(BipedWalkEnvCfg):
    """Stage 2 v2: like BipedWalk, plus a gait clock so the hind feet step left/right alternately
    (the clock-free walk learned an odd gait), and a CoM-over-feet term against stand-still drift."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})
        self.observations.critic.gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})
        hind = SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"], preserve_order=True)
        self.rewards.gait = RewTerm(
            func=bmdp.biped_feet_gait, weight=1.0,
            params={"period": GAIT_PERIOD, "offset": [0.0, 0.5], "threshold": 0.55, "sensor_cfg": hind,
                    "command_name": "base_velocity"},
        )
        self.rewards.swing_height = RewTerm(
            func=bmdp.biped_swing_height, weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["RL_foot", "RR_foot"], preserve_order=True),
                    "target_height": 0.08, "std": 0.05, "tanh_mult": 2.0, "command_name": "base_velocity"},
        )
        self.rewards.still_both_feet = RewTerm(
            func=bmdp.biped_both_feet_when_still, weight=0.3,
            params={"sensor_cfg": hind, "command_name": "base_velocity"},
        )
        self.rewards.com_over_feet = RewTerm(func=bmdp.loaded_com_over_hind_feet, weight=0.5, params={"std": 0.1})
        self.rewards.hind_air_time.weight = 0.2


@configclass
class BipedGaitWalkPlayEnvCfg(BipedGaitWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


HIND_POSE_JOINTS = ["RL_thigh_joint", "RR_thigh_joint", "RL_calf_joint", "RR_calf_joint"]
HIND_POSE_TARGET = [2.4, 2.4, -1.6, -1.6]  # knees ~0.75 rad from the straight limit (-0.84); v1 sat at -1.2..-1.4


@configclass
class BipedGaitWalkV2EnvCfg(BipedGaitWalkEnvCfg):
    """GaitWalk v2 (eval of v1, 2026-10-16): knees locked straight, left/right thigh offset 0.8 rad,
    weak/asymmetric turning (0.8 cmd -> L 0.53 / R 0.31), almost no backing up, 1.5 m drift at zero cmd."""

    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        # knees: bent symmetric pose prior + height target instead of "taller is better" (ramp saturated at 0.5)
        r.hind_pose = RewTerm(func=bmdp.hind_joint_pose_l2, weight=-0.5,
                              params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIND_POSE_JOINTS, preserve_order=True),
                                      "target": HIND_POSE_TARGET})
        r.height = RewTerm(func=bmdp.height_target_exp, weight=1.0, params={"target": 0.50, "std": 0.05})
        # tracking: tighter kernels (std^2 0.25 -> 0.15) so standing still / under-turning costs more
        r.track_lin_vel.params["std"] = math.sqrt(0.15)
        r.track_yaw_rate.params["std"] = math.sqrt(0.15)
        r.track_yaw_rate.weight = 1.5
        r.still_drift = RewTerm(func=bmdp.biped_drift_when_still, weight=-4.0, params={"command_name": "base_velocity"})
        # swing lasts 0.45 * 0.6 = 0.27 s, so a 0.3 s air-time threshold could never pay out
        r.hind_air_time.params["threshold"] = 0.25


@configclass
class BipedGaitWalkV2PlayEnvCfg(BipedGaitWalkV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class BipedGaitWalkV3EnvCfg(BipedGaitWalkEnvCfg):
    """GaitWalk v3: v2's tracking / turning / drift fixes, but knee bend only in swing (user, 2026-10-16:
    the goal is clearing the ground by flexing the swing knee, not walking crouched) -> no static pose prior,
    v1 height ramp kept."""

    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        r.track_lin_vel.params["std"] = math.sqrt(0.15)
        r.track_yaw_rate.params["std"] = math.sqrt(0.15)
        r.track_yaw_rate.weight = 1.5
        r.still_drift = RewTerm(func=bmdp.biped_drift_when_still, weight=-4.0, params={"command_name": "base_velocity"})
        r.hind_air_time.params["threshold"] = 0.25
        # mid-swing knee flexion: 1.0 rad more bend than the straight limit (calf -0.84 -> about -1.84)
        r.swing_knee_flex = RewTerm(
            func=bmdp.hind_swing_knee_flex, weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RL_calf_joint", "RR_calf_joint"], preserve_order=True),
                    "period": GAIT_PERIOD, "offset": [0.0, 0.5], "stance_frac": 0.55,
                    "straight_calf": -0.84, "flex_target": 1.0, "command_name": "base_velocity"},
        )


@configclass
class BipedGaitWalkV3PlayEnvCfg(BipedGaitWalkV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class BipedGaitWalkV4EnvCfg(BipedGaitWalkEnvCfg):
    """GaitWalk v4 (user, 2026-10-16: "knees either way; walk correctly, cleanly and fast"). Built on v1
    (best overall so far); v2/v3 knee terms dropped. Additions follow unitree_rl_lab's G1 / Go2 recipes:
    command-range curriculum toward faster speeds, step-timing symmetry, clearance 0.1 m, no vertical bounce,
    smoother actions, stronger yaw tracking, no drift at zero command."""

    def __post_init__(self):
        super().__post_init__()
        # -- fast: grow the command ranges while tracking is > 80 % of its weight (Go2 lin_vel_cmd_levels)
        old = self.commands.base_velocity
        self.commands.base_velocity = mdp.UniformLevelVelocityCommandCfg(
            asset_name=old.asset_name, resampling_time_range=old.resampling_time_range,
            rel_standing_envs=old.rel_standing_envs, rel_heading_envs=0.0, heading_command=False, debug_vis=False,
            ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.8, 0.8)),
            limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.2, 1.2)),
        )
        self.curriculum = CurriculumV4Cfg()
        r = self.rewards
        # -- correct: turning (v1 reached 38-67 % of 0.8 rad/s), no drift when told to stop
        r.track_yaw_rate.weight = 1.0
        r.still_drift = RewTerm(func=bmdp.biped_drift_when_still, weight=-2.0, params={"command_name": "base_velocity"})
        # -- clean: symmetric step timing (Go2), clearance 0.1 m / feet_slide -0.2 / action_rate -0.05 (G1), no bounce
        hind = SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"], preserve_order=True)
        r.air_time_variance = RewTerm(func=mdp.air_time_variance_penalty, weight=-1.0, params={"sensor_cfg": hind})
        r.swing_height.weight = 1.0
        r.swing_height.params["target_height"] = 0.1
        r.feet_slide.weight = -0.2
        r.action_rate.weight = -0.05
        r.vertical_bounce = RewTerm(func=bmdp.lin_vel_z_world_l2, weight=-1.0)
        r.hind_air_time.params["threshold"] = 0.25  # swing is 0.27 s; 0.3 could never pay out


@configclass
class CurriculumV4Cfg:
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels, params={"reward_term_name": "track_lin_vel"})
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels, params={"reward_term_name": "track_yaw_rate"})


@configclass
class BipedGaitWalkV4PlayEnvCfg(BipedGaitWalkV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class BipedGaitWalkV5EnvCfg(BipedGaitWalkV4EnvCfg):
    """GaitWalk v5: v4's rewards, trained FROM SCRATCH with left-right symmetry augmentation (agents.py,
    symmetry.py). Every warm start (stand -> v1 -> v4) carried a 0.7 rad L/R hip-pitch offset and the policy
    curved right on straight commands. Commands start small and grow with the curriculum, like unitree_rl_lab's
    Go2 default (initial +-0.1 m/s); yaw starts at go2_quad2hand's biped range (+-0.5 rad/s)."""

    def __post_init__(self):
        super().__post_init__()
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.1, 0.1), (-0.1, 0.1), (-0.5, 0.5)


@configclass
class BipedGaitWalkV5PlayEnvCfg(BipedGaitWalkV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class BipedGaitWalkV6EnvCfg(BipedGaitWalkV5EnvCfg):
    """GaitWalk v6 (eval of v5, 2026-10-17): symmetric and straight, but turned 12-17 % of the command and
    walked 55-65 % of the commanded speed. Per-step torso twist (0.7-1.1 rad/s) made the instantaneous yaw-rate
    reward flat in the mean turn rate -> track an integrated heading target instead (policy observes the
    heading error), and tighten the lin-vel kernel (std^2 0.25 -> 0.1: a 45 % speed shortfall cost only 12 %).
    Warm start from v5 model_9999 with scripts/warmstart.py (+1 actor / +1 critic input)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.heading_target = EventTerm(
            func=bmdp.update_heading_target, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=True,
            params={"command_name": "base_velocity", "leash": 0.5},
        )
        self.observations.policy.heading_error = ObsTerm(func=bmdp.heading_error)
        self.observations.critic.heading_error = ObsTerm(func=bmdp.heading_error)
        r = self.rewards
        r.track_heading = RewTerm(func=bmdp.track_heading_exp, weight=1.5, params={"std": 0.25})
        r.track_yaw_rate.weight = 0.25
        r.track_lin_vel.params["std"] = math.sqrt(0.1)
        self.curriculum.ang_vel_cmd_levels.params["reward_term_name"] = "track_heading"


@configclass
class BipedGaitWalkV6PlayEnvCfg(BipedGaitWalkV6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class BipedGaitWalkV6bEnvCfg(BipedGaitWalkV6EnvCfg):
    """v6b: v6 resumed at 12000. The lin-vel curriculum stalled at +-0.3 m/s (tracking 72-77 % < Go2's hard-coded
    80 %, which assumes std^2 0.25) -> threshold 0.6 (same tracking quality at std^2 0.1). Command ranges start
    where v6's curriculum had reached."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.lin_vel_cmd_levels = CurrTerm(
            func=bmdp.lin_vel_cmd_levels_thr, params={"reward_term_name": "track_lin_vel", "threshold": 0.6})
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.3, 0.3), (-0.3, 0.3), (-1.2, 1.2)


@configclass
class BipedGaitWalkV6bPlayEnvCfg(BipedGaitWalkV6bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
