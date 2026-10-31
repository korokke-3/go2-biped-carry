"""Tripod (2026-10-26): Go2 on three legs with the right front leg lifted - hopping ("けんけん") and walking with a bag
hung on the lifted forearm (user request, 2026-10-26). One policy, two gait modes (tripod_mdp).

Built from go2_quad2hand's QUADRUPED mode (ref = refs/go2_quad2hand, go2_quad2hand_config.py / go2_quad2hand.py) with
one leg turned into a free leg the way the ref's biped ("hand") mode treats its free legs. Item by item (ref -> here):
  action scale 0.25, PD 30 / 0.75 (knee curve x1.5 as RefWalk-V2), decimation 4 .. same
  commands quad vx +-1.0, vy +-0.8, wz +-1.0, deadzone 0.1, 10 s ........ vx -0.3..0.5, vy +-0.2, wz +-0.6 + 25 %
                                                                          standing envs: three legs, about half the
                                                                          quad range; standing is a goal here
  gait_mode one-hot + per-mode period / offsets (quad 0.5 s [0,.5,.5,0]) . hop 0.4 s (ref hand period) front vs
                                                                          hinds [0,.5,.5], stance 0.6; walk 0.6 s
                                                                          3-beat [0,1/3,2/3], stance 0.75 (one leg in
                                                                          the air at a time); clock off at zero
                                                                          command: all three feet down (user)
  tracking (heading frame) lin 1.5 / yaw 1.0, sigma 0.25 ................ same
  base_height_quad exp(-(h-0.36)^2/0.04) 1.0 ............................ same
  body_orientation_quad (pitch 0) sigma 0.04, 1.0 ....................... same
  foot_clearance_quad 0.05 m, sigma 0.01, 0.2 ............................ same (three stance feet)
  contact_no_vel -0.1, hip_close_to_default -1.0 ......................... same (hips of the stance legs)
  free legs (hand mode): contact -2.0, dof pos -0.1 to default ......... FR foot contact -2.0, dof pos -0.1 to the
                                                                          lifted pose LIFT_POSE
  collision thigh / calf -1.5, torques -1e-4, dof_vel -1e-4, dof_acc -1e-7, action_rate -1e-3, termination -200 same
  terminate on base / head / hip contact ................................. same
  DR (friction, mass, CoM, push, joint friction / armature, Kp/Kd) ...... same (RefWalk-V3) + foot torsion 0-0.03 m
                                                                          + motor strength x0.8-1.5 (HoldBall: MuJoCo's
                                                                          flat limits made the stand-up policies fall)
  obs / history 5 / PPO 512-256-128, entropy 0.01 ........................ same, + gait mode, per-mode clock
  new: com_over_support 0.5 (standing; ALaM, Shin et al. 2025), foot impact -0.01 x (F - 200 N) (legged_gym
       feet_contact_forces; hopping lands on one foot), bag_hung 1.0 + terminate when the bag slides off.
  no L/R mirror augmentation: the task is asymmetric (FR lifted).
"""
import math

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
from . import tripod_mdp as tmdp
from .refwalk_env_cfg import CMD_SCALE, HISTORY, RefWalkV3EnvCfg

STANCE = tmdp.STANCE_FEET


@configclass
class TripodObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, scale=CMD_SCALE, clip=(-100, 100),
                                    params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        gait_mode = ObsTerm(func=tmdp.gait_mode_onehot)
        gait_clock = ObsTerm(func=tmdp.gait_clock)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = HISTORY

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100))
        base_height = ObsTerm(func=bmdp.base_height, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, scale=CMD_SCALE, clip=(-100, 100),
                                    params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        gait_mode = ObsTerm(func=tmdp.gait_mode_onehot)
        gait_clock = ObsTerm(func=tmdp.gait_clock)
        actuator_kp_scale = ObsTerm(func=rmdp.actuator_kp_scale)
        actuator_kd_scale = ObsTerm(func=rmdp.actuator_kd_scale)
        joint_friction = ObsTerm(func=rmdp.joint_friction)
        joint_armature = ObsTerm(func=rmdp.joint_armature)
        bag = ObsTerm(func=tmdp.bag_obs)

        def __post_init__(self):
            self.history_length = HISTORY

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TripodRewardsCfg:
    # std^2 0.25 (ref) until chunk 4: standing still under a -0.2 m/s command kept 85 % of the term and the policy
    # never walked backwards (sideways 0.07 of 0.15); 0.1 as in HoldRef-V3 / HoldV4 from chunk 5
    tracking_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5,
                               params={"command_name": "base_velocity", "std": math.sqrt(0.1)})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0,
                               params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    gait = RewTerm(func=tmdp.tripod_gait, weight=0.5,
                   params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE, preserve_order=True)})
    base_height = RewTerm(func=rmdp.base_height_hand, weight=1.0, params={"target": 0.36, "sigma": 0.04})
    body_orientation = RewTerm(func=tmdp.base_orientation_flat, weight=1.0, params={"sigma": 0.04})
    foot_clearance = RewTerm(func=rmdp.foot_clearance_hand, weight=0.2,
                             params={"target": 0.05, "sigma": 0.01,
                                     "asset_cfg": SceneEntityCfg("robot", body_names=STANCE, preserve_order=True)})
    contact_no_vel = RewTerm(func=rmdp.contact_no_vel, weight=-0.1,
                             params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE, preserve_order=True),
                                     "asset_cfg": SceneEntityCfg("robot", body_names=STANCE, preserve_order=True)})
    hip_close_to_default = RewTerm(func=rmdp.joint_dev_l2, weight=-1.0,
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "RL_hip_joint", "RR_hip_joint"])})
    free_contact = RewTerm(func=bmdp.contact_count, weight=-2.0,
                           params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FR_foot"])})
    free_dof_pos = RewTerm(func=rmdp.joint_dev_l2, weight=-0.1,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=tmdp.FREE_JOINTS, preserve_order=True),
                                   "target": tmdp.LIFT_POSE})
    com_over_support = RewTerm(func=tmdp.com_over_support, weight=0.5, params={"std": 0.1})
    foot_impact = RewTerm(func=mdp.contact_forces, weight=-0.01,
                          params={"threshold": 200.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=STANCE)})
    bag_hung = RewTerm(func=tmdp.bag_hung, weight=1.0)
    forearm_up = RewTerm(func=tmdp.forearm_up, weight=0.5)  # from chunk 2 (walk mode lowered the paw, bags slid off)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    collision = RewTerm(func=mdp.undesired_contacts, weight=-1.5,
                        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"])})
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TripodTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", ".*_hip"]), "threshold": 1.0},
    )
    bag_lost = DoneTerm(func=tmdp.bag_lost)


@configclass
class TripodEnvCfg(RefWalkV3EnvCfg):
    observations: TripodObservationsCfg = TripodObservationsCfg()
    rewards: TripodRewardsCfg = TripodRewardsCfg()
    terminations: TripodTerminationsCfg = TripodTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity
        c.ranges = rmdp.DeadzoneVelocityCommandCfg.Ranges(lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.6, 0.6))
        c.rel_standing_envs = 0.25
        c.limit_ranges = c.ranges  # (no range curriculum; unitree_rl_lab's deploy export reads limit_ranges)
        # quadruped resets only (the ref's init: quad stance, random yaw), then mode / bag / lifted-leg init
        self.events.reset_robot.params.update({"biped_prob": 0.0, "mid_prob": 0.0})
        self.events.reset_tripod = EventTerm(
            func=tmdp.reset_tripod, mode="reset",
            params={"p_walk": 0.5, "p_bag": 0.7, "mass_range": (0.2, 2.0), "rope_range": (0.08, 0.20),
                    "hang_range_s": (0.5, 4.0), "p_lifted": 0.5})
        # slip_s 0.5 (chunk 1: 0.2) - a handle slides along the forearm, it does not drop at once
        self.events.bag = EventTerm(func=tmdp.bag_step, mode="interval", interval_range_s=(0.02, 0.02),
                                    is_global_time=True, params={"slip_s": 0.5})
        self.events.motor_strength = EventTerm(func=rmdp.randomize_motor_strength, mode="reset",
                                               params={"scale_range": (0.8, 1.5)})
        self.curriculum = None


@configclass
class TripodPlayEnvCfg(TripodEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.motor_strength = None


@configclass
class TripodHighEnvCfg(TripodEnvCfg):
    """TripodHigh (2026-10-28, user: carry the bag higher): walk mode only, the body stands taller and pitches nose-up so
    the elbow (bag) rises; the hop mode keeps the Tripod pose.
    Geometry (planar, Go2 thigh / calf 0.213 m): the FR thigh stops at -1.57 rad (upper arm horizontal), so the elbow
    cannot rise above the shoulder (Tripod c6: elbow 0.33 m, bag 0.18 m). The shoulder rises with the body height
    (0.36 -> 0.40 m: +4 cm) and the elbow, ~0.40 m in front of the base, with a nose-up pitch (10 deg: +7 cm).
    Target: elbow 0.33 -> 0.45 m, i.e. the bag ~10 cm higher.
      base_height + body_orientation (ref quad: 0.36 m, pitch 0; sigma 0.04 / 0.04) -> one per-mode term
          base_pose_by_mode: hop 0.36 m / 0 deg, walk 0.40 m / 10 deg nose-up, weight 2.0 (= the two weights 1.0)
      elbow_height (walk): 0 at 0.33 m -> 1 at 0.45 m, weight 1.0
      free_dof_pos target thigh -1.4 -> -1.5 (upper arm closer to horizontal, still 0.07 rad inside the limit)
    Warm start from Tripod c6 (model_8994), +3000 iterations (two chunks, the Tripod std^2 change took one)."""

    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        r.base_height = None
        r.body_orientation = RewTerm(func=tmdp.base_pose_by_mode, weight=2.0,
                                     params={"pitch": [0.0, math.radians(10.0)], "height": [0.36, 0.40],
                                             "sigma_pitch": 0.04, "sigma_height": 0.04})
        r.elbow_height = RewTerm(func=tmdp.elbow_height, weight=1.0, params={"low": 0.33, "high": 0.45})
        r.free_dof_pos.params["target"] = [0.0, -1.5, -1.2]


@configclass
class TripodHighPlayEnvCfg(TripodHighEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.motor_strength = None


@configclass
class TripodCoolEnvCfg(TripodHighEnvCfg):
    """TripodCool (2026-10-28): TripodHigh's high carry without the motor load it bought.
    TripodHigh c2 per joint (walk, 2 kg, 0.2 m/s): the lifted FR thigh sat on its stop (-1.56, limit -1.571) at 20.5 Nm
    RMS (0.86 of peak; Tripod c6 the same, also with no bag), the FL knee was locked straight on its stop (-0.84,
    limit -0.838) at 12-14 Nm, RR knee 13 Nm vs RL 6 Nm, peak foot force 330 N (c6 226 N). Pushing on a stop buys
    ~1 cm of elbow height (0.06 rad x 0.213 m) and costs heat: holding the leg up needs a few Nm.
      motor_heat      sum over all 12 motors (tau / peak)^2, weight -1.0 (RefWalk-V4 / HoldV4 hind_load final weight;
                      all joints since a penalty on some would shift the load to the others)
      motor_heat_max  max over the motors (tau / peak)^2, weight -1.0 (the hottest motor trips first; same scale)
      limit_margin    -10 (unitree_rl_lab Go2 dof_pos_limits) inside the last 0.15 rad of the calves (RefWalk-V4,
                      the locked-knee exploit) and 0.05 rad of hips / thighs (the FR arm target -1.5 stays outside)
    Height rewards unchanged (user: keep the high carry). Warm start from Tripod c6 (model_8994), +3000."""

    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        r.motor_heat = RewTerm(func=tmdp.motor_heat, weight=-1.0)
        r.motor_heat_max = RewTerm(func=tmdp.motor_heat_max, weight=-1.0)
        r.limit_margin_calf = RewTerm(func=rmdp.joint_limit_margin, weight=-10.0,
                                      params={"margin": 0.15, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])})
        r.limit_margin_hip = RewTerm(func=rmdp.joint_limit_margin, weight=-10.0,
                                     params={"margin": 0.05, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint"])})


@configclass
class TripodCoolPlayEnvCfg(TripodCoolEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.motor_strength = None


STANCE_HIND = tmdp.stance_feet_for("RR")
# lifted RR pose (planar kinematics, thigh = calf = 0.213 m, base 0.32 m): the paw is above the knee only for thigh >= 2.8
# with the calf near straight; thigh 3.0 (thigh ~ straight up, knee 3 cm behind the hip) / calf -1.0 puts the paw
# 9 cm above the knee, knee 0.53 m, bag ~0.39 m (rope 0.14); hip +0.5 swings the raised knee outward, away from the
# trunk (for an upward thigh the abduction sign is the reverse of a standing leg's; fixed-base diag: +0.3 hung the bag only
# 10 cm out from the hip joint at hip height - the real bag would rub the hip housing)
LIFT_HIND = [0.5, 3.0, -1.0]


@configclass
class TripodHindEnvCfg(TripodCoolEnvCfg):
    """TripodHind (2026-10-28, user: try carrying with a hind leg - maybe easier to hold): RR lifted, stance FL, FR, RL.
    The rear thigh swings up past horizontal (limit 4.54 rad) while the front one stops at horizontal (-1.571), so the
    knee where the bag hangs can rise above the hip: knee 0.53 m vs TripodHigh's elbow 0.46 m.
    Same terms and weights as TripodCool (heat sum / max, limit margins), with the leg names swapped:
      stance terms (gait, clearance, contact_no_vel, foot impact) on RL, FL, FR; the hop pairs RL against FL + FR
      hip_close_to_default on FL / FR / RL hips; free contact -2 on RR_foot; free dof prior -0.1 to LIFT_HIND
      body pose both modes 0.36 m / pitch 0 (a nose-up pitch lowers the rear); elbow (knee) height 0 at 0.40 m -> 1 at
      0.53 m (LIFT_HIND geometry)
    Expected cost (to be measured): the lone stance hind leg carries the whole rear share (Tripod front carry: two hind
    legs, RR knee 13 / RL 6 Nm).
    Warm start from Tripod c6 (model_8994), +6000 (four 1500 chunks; Tripod from scratch passed after 9000)."""

    def __post_init__(self):
        super().__post_init__()
        r = self.rewards
        for t in (r.gait, r.contact_no_vel, r.foot_impact):
            t.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=STANCE_HIND, preserve_order=True)
        for t in (r.foot_clearance, r.contact_no_vel):
            t.params["asset_cfg"] = SceneEntityCfg("robot", body_names=STANCE_HIND, preserve_order=True)
        r.hip_close_to_default.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint"])
        r.free_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=["RR_foot"])
        r.free_dof_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"], preserve_order=True)
        r.free_dof_pos.params["target"] = LIFT_HIND
        r.body_orientation.params.update({"pitch": [0.0, 0.0], "height": [0.36, 0.36]})
        r.elbow_height.params.update({"low": 0.40, "high": 0.53})
        self.events.reset_tripod.params.update({"free_leg": "RR", "lift_pose": LIFT_HIND})


@configclass
class TripodHindPlayEnvCfg(TripodHindEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.motor_strength = None


@configclass
class TripodRealCurriculumCfg:
    # tracking thresholds as fractions of the weights: lin kernel std^2 0.1 since Tripod chunk 5 -> 0.55 (HoldV4's
    # mapping of RefWalk's 0.8 at std^2 0.25), yaw kernel std^2 0.25 -> RefWalk-V4's 0.7
    cmd_ranges = CurrTerm(func=rmdp.cmd_range_curriculum,
                          params={"lin_term": "tracking_lin_vel", "ang_term": "tracking_ang_vel",
                                  "lin_threshold": 0.55, "ang_threshold": 0.7})


@configclass
class TripodRealEnvCfg(TripodCoolEnvCfg):
    """TripodReal (2026-10-31, user: fix what the Tripod lacks): TripodCool + the measured real joints + a wider start
    + faster commands. On the robot (2026-10-30) TripodCool c2 stood on three legs for 33 s; in MuJoCo with the measured
    joints it slowed 0.20 -> 0.06-0.09 m/s and its walk gait tipped over 3 of 6 runs; a second start from displaced feet
    tipped over on the robot (deploy/README.md).
      measured joints ...... switch_env_cfg.add_real_joint_terms (gearbox friction Fs + mu|tau| + Fd x U(0.5, 1.5),
                             knee rotor inertia 0.03-0.08, base mass -1..+2 kg; motor strength x0.8-1.5 already here)
      start ................ quadruped resets spread the joints +-0.15 rad (0.05; the robot ran from 0.14, fell from 0.30)
      commands ............. grow from vx -0.3..0.5 / vy +-0.2 / wz +-0.6 by RefWalk-V4's curriculum (half of the draws
                             stay in the old range) up to the ref's quadruped range (vx +-1.0, wz +-1.0) with backwards
                             -0.6 and sideways +-0.5 (vy ref 0.8; three legs)
    Warm start from TripodCool c2 (model_11299), 3 x 1500 (switch / Hold chunks)."""

    def __post_init__(self):
        super().__post_init__()
        from .switch_env_cfg import add_real_joint_terms
        add_real_joint_terms(self)
        self.events.reset_robot.params["quad_joint_noise"] = 0.15
        c = self.commands.base_velocity
        c.base_ranges = rmdp.DeadzoneVelocityCommandCfg.Ranges(lin_vel_x=(-0.3, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.6, 0.6))
        c.p_base = 0.5
        c.limit_ranges = rmdp.DeadzoneVelocityCommandCfg.Ranges(lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0))
        self.curriculum = TripodRealCurriculumCfg()


@configclass
class TripodRealPlayEnvCfg(TripodRealEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.motor_strength = None
        self.events.joint_friction_real.params["scale_range"] = (1.0, 1.0)  # measured values
        self.events.reset_robot.params["quad_joint_noise"] = 0.05
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.p_base = 0.0
