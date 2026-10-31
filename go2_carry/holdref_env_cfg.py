"""HoldRef: receive / hold / carry real objects, built on RefWalk (go2_quad2hand-aligned walking, 2026-10-20).

Object mechanics, curriculum and object rewards are those of Hold (hold_env_cfg.py / box_mdp.py). Changes:
* robot / gait / regularization: RefWalk's (the base policy is warm-started from it).
* hand-over (task: unknown objects, put down slowly or suddenly, possibly taken away again):
  - gentle hand-overs are released slowly (hand support m g -> 0 over 0-2 s); the other half is dropped 0-10 cm at once
  - placement noise: 0-3 cm further from the belly, +-3 cm sideways, +-10 deg yaw
  - 30 % of the held objects are taken away 3-10 s after placement (lifted over 0-1 s), then the episode goes on
  - the hand-over waits for the arms only while the assist curriculum is > 0 (afterwards the person decides)
* contacts: the front legs touch the object, so every ground test uses ground-filtered sensors (GROUND_COLLIDER):
  falls = base / head / hip on the ground (RefWalk's termination set); front feet "contact" -> front feet below
  0.2 m (Hold); collision penalty and contact_no_vel on the hind legs only (the front legs' contacts are the object).
* knees: penalty only (-1.5, RefWalk); no knee termination (user, 2026-10-20: briefly touching and getting up is ok).
* tray-pose terms: RefWalk's front_dof_pos_hand (L2, -0.1) only while NOT holding; Hold's ready_pose (L1, -1.0) while
  waiting / in tray-walk episodes; fold_arms (+0.3) while holding.
"""
import json
import math
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import box_mdp as hmdp
from . import mdp as bmdp
from . import refwalk_mdp as rmdp
from .hold_env_cfg import GROUND_COLLIDER, HoldSceneCfg
from .refwalk_env_cfg import FRONT_LEG_JOINTS, KNEE_RATIO, TRAY_POSE, RefWalkEnvCfg, refwalk_actuators

HIP_GROUND = ["fl_hip_ground", "fr_hip_ground", "rl_hip_ground", "rr_hip_ground"]


def front_dof_pos_not_holding(env, asset_cfg: SceneEntityCfg, target: list[float]):
    """RefWalk's front_dof_pos_hand, off while an object is held (the arms then fold around it)."""
    return rmdp.joint_dev_l2(env, asset_cfg, target) * (~hmdp.active(env)).float()


@configclass
class HoldRefCurriculumCfg:
    object = CurrTerm(func=hmdp.object_curriculum, params={"start": None})  # Hold's curriculum


@configclass
class HoldRefSceneCfg(HoldSceneCfg):
    fl_hip_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/FL_hip", filter_prim_paths_expr=[GROUND_COLLIDER])
    fr_hip_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/FR_hip", filter_prim_paths_expr=[GROUND_COLLIDER])
    rl_hip_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/RL_hip", filter_prim_paths_expr=[GROUND_COLLIDER])
    rr_hip_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/RR_hip", filter_prim_paths_expr=[GROUND_COLLIDER])

    def __post_init__(self):
        super().__post_init__()
        refwalk_actuators(self.robot, KNEE_RATIO)  # RefWalk v2 (gains 30 / 0.75, knee curve x1.5)


def add_hold_terms(self):
    """Object, hand-over, curriculum, critic object terms, hold rewards and ground-only terminations of HoldRef
    (shared by HoldRef and HoldV4)."""
    # ---- object (Hold)
    self.events.object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("object"), "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (0.25, 0.9), "restitution_range": (0.0, 0.1), "num_buckets": 64},
    )
    self.events.object_mass = EventTerm(func=hmdp.object_mass_curriculum, mode="reset")
    self.events.object_handover = EventTerm(
        func=hmdp.object_handover, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=True,
        params={"p_object": 0.8, "p_tray_walk": 1.0, "ready_range_s": (0.5, 3.0), "max_wait_s": 3.0,
                "ready_tol": 0.6, "p_gentle": 0.5, "drop_max": 0.10,
                "release_range_s": (0.0, 2.0), "place_noise": (0.03, 0.03, math.radians(10.0)),
                "p_remove": 0.3, "remove_after_s": (3.0, 10.0), "remove_ramp_s": (0.0, 1.0),
                "ready_gate_while_assist": True},
    )
    # HOLD_CURR_START='{"level": 0, "m_max": 0.3, "assist": 0.2}' restores the curriculum state on a resumed run
    self.curriculum = HoldRefCurriculumCfg()
    self.curriculum.object.params["start"] = json.loads(os.environ.get("HOLD_CURR_START", "null"))
    # critic: object state (appended after RefWalk's terms -> scripts/warmstart.py --critic_new)
    c = self.observations.critic
    c.obj_rel_pos = ObsTerm(func=hmdp.obj_rel_pos)
    c.obj_rel_vel = ObsTerm(func=hmdp.obj_rel_vel)
    c.obj_mass = ObsTerm(func=hmdp.obj_mass)
    c.obj_present = ObsTerm(func=hmdp.obj_present)
    c.obj_dims = ObsTerm(func=hmdp.obj_dims)
    # ---- rewards
    r = self.rewards
    r.holding = RewTerm(func=hmdp.holding, weight=1.5)
    r.obj_shake = RewTerm(func=hmdp.obj_shake, weight=-0.5)
    r.obj_tilt = RewTerm(func=hmdp.obj_tilt, weight=-1.0)
    r.fold_arms = RewTerm(func=hmdp.fold_arms, weight=0.3)
    r.ready_pose = RewTerm(func=hmdp.ready_pose, weight=-1.0)
    r.obj_squeeze = RewTerm(func=hmdp.obj_squeeze, weight=-1.0, params={"sensor_name": "object_contact", "factor": 1.5})
    r.com_over_feet = RewTerm(func=hmdp.com_over_feet_with_obj, weight=0.5, params={"std": 0.1})
    r.front_contact_hand = RewTerm(
        func=bmdp.feet_below, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot"]), "height": 0.2})
    r.front_dof_pos_hand = RewTerm(
        func=front_dof_pos_not_holding, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=FRONT_LEG_JOINTS, preserve_order=True),
                "target": TRAY_POSE})
    hind = ["RL_foot", "RR_foot"]
    r.contact_no_vel.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=hind, preserve_order=True)
    r.contact_no_vel.params["asset_cfg"] = SceneEntityCfg("robot", body_names=hind, preserve_order=True)
    r.collision.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=["R.*_thigh", "R.*_calf"])
    # ---- terminations: falls on the GROUND only (the object touches the front legs / belly); drop ends the episode
    self.terminations.body_contact = DoneTerm(
        func=bmdp.body_touches_ground,
        params={"sensor_names": ["base_ground", "head_upper_ground", "head_lower_ground"] + HIP_GROUND})
    self.terminations.object_dropped = DoneTerm(func=hmdp.obj_dropped, params={"min_height": 0.3, "max_dist": 0.5})


@configclass
class HoldRefEnvCfg(RefWalkEnvCfg):
    scene: HoldRefSceneCfg = HoldRefSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        add_hold_terms(self)


@configclass
class HoldRefPlayEnvCfg(HoldRefEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None


@configclass
class HoldRefV2EnvCfg(HoldRefEnvCfg):
    """+ per-env foot torsional friction (RefWalk-V3): PhysX radius 0.03 m turned RefWalk-v2 at 0.28 of 0.5 rad/s,
    the same as MuJoCo's condim-4 feet (0.27), vs 0.50 without."""

    def __post_init__(self):
        super().__post_init__()
        from .refwalk_env_cfg import TORSION_RANGE
        self.events.foot_torsion = EventTerm(func=rmdp.randomize_foot_torsional_friction, mode="prestartup",
                                             params={"radius_range": TORSION_RANGE})


@configclass
class HoldRefV2PlayEnvCfg(HoldRefV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None


@configclass
class HoldRefV3EnvCfg(HoldRefV2EnvCfg):
    """holdref_v4 (V2 = v3 + torsion 0-0.04 m) degraded from the first 200 iterations: a -0.1 m/s backward drift at
    zero command and 2-3x more drops, while learning to turn on torsional feet. The ref's lin-vel kernel (sigma^2 0.25)
    charges a 0.1 m/s drift only 4 % -> sigma^2 0.1 (GaitWalk v6 used it for the same reason), and the torsion range
    is narrowed to 0-0.03 m (0.03 reproduced MuJoCo's turn-rate drop)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.tracking_lin_vel.params["std"] = math.sqrt(0.1)
        self.events.foot_torsion.params["radius_range"] = (0.0, 0.03)


@configclass
class HoldRefV3PlayEnvCfg(HoldRefV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
