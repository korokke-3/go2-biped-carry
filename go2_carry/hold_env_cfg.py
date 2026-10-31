"""Hold stage 1: stand and receive a real object on the front legs, keep it (see box_mdp.py for the mechanics).

Built on GaitWalk v6b (symmetric walking, no virtual-load habits) + what the Load-V2..V7 runs taught:
3-step proprioceptive history, lateral-only tracking term, mirror loss, low entropy (agent cfg), wider friction.
80 % of the episodes are object episodes (zero command); the rest walk with the arms held in the tray pose, so
walking is kept. One consistent rule for the policy, which cannot see the episode type: without an object hold the
tray pose (always ready to receive), with an object fold the arms as far as the object allows.
"""
import json
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
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
from .biped_env_cfg import BipedGaitWalkV6bEnvCfg, BipedSceneCfg

# the plane terrain's collider (a filter on "/World/ground" matched nothing: every ground-only sensor, i.e. the
# fall and knee terminations, read 0 until 2026-10-20 - found when knee_on_ground stayed 0 while the shins lay flat)
GROUND_COLLIDER = "/World/ground/terrain/GroundPlane/CollisionPlane"
_BROWN = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.52, 0.3))
STEP_PERIOD = 0.4  # s, gait clock period (go2_quad2hand biped mode)


def _shape_cfg(dims, k, c):
    return sim_utils.CuboidCfg(
        size=dims, visual_material=_BROWN, activate_contact_sensors=True,  # for the squeeze-guard sensor
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6,
                                                        compliant_contact_stiffness=k, compliant_contact_damping=c),
    )


@configclass
class HoldSceneCfg(BipedSceneCfg):
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[_shape_cfg(d, k, c) for d, k, c, _ in hmdp.BOX_SHAPES],
            random_choice=False,  # env i -> shape i % K (known per env)
            activate_contact_sensors=True,  # the multi-asset value overrides the per-shape one
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )
    # object <-> front legs, per leg body (for the squeeze guard)
    object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/FL_calf", "{ENV_REGEX_NS}/Robot/FR_calf",
                                "{ENV_REGEX_NS}/Robot/FL_thigh", "{ENV_REGEX_NS}/Robot/FR_thigh"])
    base_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base", filter_prim_paths_expr=[GROUND_COLLIDER])
    rl_thigh_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/RL_thigh", filter_prim_paths_expr=[GROUND_COLLIDER])
    rr_thigh_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/RR_thigh", filter_prim_paths_expr=[GROUND_COLLIDER])
    rl_calf_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/RL_calf", filter_prim_paths_expr=[GROUND_COLLIDER])
    rr_calf_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/RR_calf", filter_prim_paths_expr=[GROUND_COLLIDER])
    head_upper_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Head_upper", filter_prim_paths_expr=[GROUND_COLLIDER])
    head_lower_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Head_lower", filter_prim_paths_expr=[GROUND_COLLIDER])


@configclass
class HoldEnvCfg(BipedGaitWalkV6bEnvCfg):
    scene: HoldSceneCfg = HoldSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        # ---- from Load-V2..V7
        self.observations.policy.history_length = 3
        self.rewards.track_lin_vel_y = RewTerm(func=bmdp.track_biped_lin_vel_y_exp, weight=1.0,
                                               params={"command_name": "base_velocity", "std": 0.02 ** 0.5})
        # friction: go2_quad2hand's 0.2-1.25 (polished floors ~0.2); objects 0.3-1.0
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
        # ---- step in place (point feet): go2_quad2hand's biped mode uses a 0.4 s gait period and rewards the gait
        # clock at all times, also at zero command. With "both feet down when still" (v6b) the holding policies knelt
        # for a third support point (hold_v8 48 % of the time, carry_walk_v3 35 %).
        for g in (self.observations.policy, self.observations.critic):
            g.gait_phase.params["period"] = STEP_PERIOD
        self.rewards.gait.params["period"] = STEP_PERIOD
        self.rewards.gait.params["command_name"] = None
        self.rewards.still_both_feet = None
        self.rewards.hind_air_time.params["threshold"] = 0.15  # swing 0.45 * 0.4 = 0.18 s
        # ---- object
        self.events.object_friction = EventTerm(
            func=mdp.randomize_rigid_body_material, mode="startup",
            params={"asset_cfg": SceneEntityCfg("object"), "static_friction_range": (0.3, 1.0),
                    "dynamic_friction_range": (0.25, 0.9), "restitution_range": (0.0, 0.1), "num_buckets": 64},
        )
        self.events.object_mass = EventTerm(func=hmdp.object_mass_curriculum, mode="reset")
        self.events.object_handover = EventTerm(
            func=hmdp.object_handover, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=True,
            params={"p_object": 0.8, "p_tray_walk": 1.0, "ready_range_s": (0.5, 2.0), "max_wait_s": 3.0,
                    "ready_tol": 0.6, "p_gentle": 0.5, "drop_max": 0.10},
        )
        # HOLD_CURR_START='{"level": 0, "m_max": 0.3, "assist": 0.2}' restores the curriculum state on a resumed run
        self.curriculum.object = CurrTerm(func=hmdp.object_curriculum,
                                          params={"start": json.loads(os.environ.get("HOLD_CURR_START", "null"))})
        c = self.observations.critic
        c.obj_rel_pos = ObsTerm(func=hmdp.obj_rel_pos)
        c.obj_rel_vel = ObsTerm(func=hmdp.obj_rel_vel)
        c.obj_mass = ObsTerm(func=hmdp.obj_mass)
        c.obj_present = ObsTerm(func=hmdp.obj_present)
        c.obj_dims = ObsTerm(func=hmdp.obj_dims)
        r = self.rewards
        r.holding = RewTerm(func=hmdp.holding, weight=1.5)
        r.obj_shake = RewTerm(func=hmdp.obj_shake, weight=-0.5)
        r.obj_tilt = RewTerm(func=hmdp.obj_tilt, weight=-1.0)
        r.fold_arms = RewTerm(func=hmdp.fold_arms, weight=0.3)
        # -0.3 left the arms ~1.8 rad (L1) off the tray pose after 300 iterations (hold_v1); -1.0 = G1 leg joint deviation
        r.ready_pose = RewTerm(func=hmdp.ready_pose, weight=-1.0)
        r.obj_squeeze = RewTerm(func=hmdp.obj_squeeze, weight=-1.0, params={"sensor_name": "object_contact", "factor": 1.5})
        r.com_over_feet = RewTerm(func=hmdp.com_over_feet_with_obj, weight=0.5, params={"std": 0.1})
        # front feet: keep them up (height), since touching the object must not count like touching the ground
        r.front_feet_off_ground = RewTerm(
            func=bmdp.feet_below, weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot"]), "height": 0.2})
        r.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["Head_.*", ".*_hip", "R.*_thigh", "R.*_calf"])
        # ---- terminations: falls = base / head on the GROUND only; dropping the object ends the episode
        self.terminations.body_contact = DoneTerm(
            func=bmdp.body_touches_ground, params={"sensor_names": ["base_ground", "head_upper_ground", "head_lower_ground"]})
        self.terminations.knee_on_ground = DoneTerm(
            func=hmdp.knee_on_ground,
            params={"sensor_names": ["rl_thigh_ground", "rr_thigh_ground", "rl_calf_ground", "rr_calf_ground"]})
        self.terminations.object_dropped = DoneTerm(func=hmdp.obj_dropped, params={"min_height": 0.3, "max_dist": 0.5})


@configclass
class HoldPlayEnvCfg(HoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
