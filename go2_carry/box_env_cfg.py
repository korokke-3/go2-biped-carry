"""Box task: a real rigid box handed onto the forearms, carried by supporting it from below (no squeezing).

Built on Load-V7 (symmetric walking with heading / lateral tracking). The virtual forearm force is replaced by a
physical box: 0.32-0.40 m wide (spans both forearms, ~0.28 m apart in the tray pose), 0.15-0.25 m deep,
0.08-0.16 m tall, 0.1-2 kg. It is placed onto the arms 0.5-4 s into the episode, half of the time gently and
otherwise dropped from up to 12 cm; 15 % of the episodes have no box. The actor keeps proprioception only
(same inputs as Load-V7); the critic additionally sees the box state.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import mdp as bmdp
from .biped_env_cfg import BipedSceneCfg
from .load_env_cfg import LoadV7EnvCfg

_BOX_PHYS = dict(
    rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6),
)
# the plane terrain's collider (a filter on "/World/ground" matched nothing: every ground-only sensor, i.e. the
# fall and knee terminations, read 0 until 2026-10-20 - found when knee_on_ground stayed 0 while the shins lay flat)
GROUND_COLLIDER = "/World/ground/terrain/GroundPlane/CollisionPlane"
_BROWN = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.52, 0.3))
# (fwd, lateral, height) in the box frame; placed yaw-aligned with the robot heading
BOX_SIZES = [(0.20, 0.36, 0.12), (0.15, 0.32, 0.08), (0.25, 0.40, 0.16), (0.18, 0.34, 0.14), (0.22, 0.38, 0.10)]


@configclass
class BoxSceneCfg(BipedSceneCfg):
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.CuboidCfg(size=s, visual_material=_BROWN, **_BOX_PHYS) for s in BOX_SIZES],
            random_choice=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )
    # fall detection against the ground only (one filtered sensor per body: multi-body + filter is unsupported)
    base_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base", filter_prim_paths_expr=[GROUND_COLLIDER])
    head_upper_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Head_upper", filter_prim_paths_expr=[GROUND_COLLIDER])
    head_lower_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Head_lower", filter_prim_paths_expr=[GROUND_COLLIDER])


@configclass
class BoxEnvCfg(LoadV7EnvCfg):
    scene: BoxSceneCfg = BoxSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        # the virtual forearm force is gone (its critic input stays, reading 0, so Load-V7 warm-starts directly)
        self.events.forearm_load = None
        self.events.box_handover = EventTerm(
            func=bmdp.box_handover, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=True,
            params={"p_box": 0.85, "place_range_s": (0.5, 4.0), "p_gentle": 0.5, "drop_range": (0.0, 0.12)},
        )
        self.events.box_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="reset",
            params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": (0.1, 2.0), "operation": "abs"},
        )
        self.events.box_friction = EventTerm(
            func=mdp.randomize_rigid_body_material, mode="startup",
            params={"asset_cfg": SceneEntityCfg("object"), "static_friction_range": (0.3, 1.0),
                    "dynamic_friction_range": (0.25, 0.9), "restitution_range": (0.0, 0.1), "num_buckets": 64},
        )
        # robot friction 0.4-1.2 -> 0.2-1.25 (go2_quad2hand's range; polished indoor floors are ~0.2, and the
        # box-forearm contact averages the two materials). Not RMA's 0.05-4.5: no adaptation module here.
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.25)
        c = self.observations.critic
        c.box_rel_pos = ObsTerm(func=bmdp.box_rel_pos)
        c.box_rel_vel = ObsTerm(func=bmdp.box_rel_vel)
        c.box_mass = ObsTerm(func=bmdp.box_mass_obs)
        c.box_present = ObsTerm(func=bmdp.box_present_obs)
        r = self.rewards
        r.box_on_forearms = RewTerm(func=bmdp.box_on_forearms, weight=2.0, params={"std": 0.08})
        r.box_tilt = RewTerm(func=bmdp.box_tilt, weight=-2.0)
        r.tray_pose = RewTerm(
            func=bmdp.front_tray_pose_l1, weight=-0.3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FR_hip_joint", "FL_thigh_joint",
                                                                       "FR_thigh_joint", "FL_calf_joint", "FR_calf_joint"],
                                                preserve_order=True),
                    "target": [0.0, 0.0, 1.3, 1.3, -1.3, -1.3]},
        )
        r.com_over_feet = RewTerm(func=bmdp.com_over_feet_with_box, weight=0.5, params={"std": 0.1})
        # the box may rest against the belly / be held by the front legs: only the ground counts as a fall, and
        # front-leg contacts (with the box) are not "undesired"
        self.terminations.body_contact = DoneTerm(
            func=bmdp.body_touches_ground, params={"sensor_names": ["base_ground", "head_upper_ground", "head_lower_ground"]})
        r.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["Head_.*", ".*_hip", "R.*_thigh", "R.*_calf"])
        self.terminations.box_dropped = DoneTerm(func=bmdp.box_dropped,
                                                 params={"max_horizontal": 0.25, "max_below": 0.10})


@configclass
class BoxPlayEnvCfg(BoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
