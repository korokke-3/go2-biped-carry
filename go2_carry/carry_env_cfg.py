"""Stage 3: stand/walk on the hind legs while holding an object between the front legs.

Objects: boxes / cylinder / sphere (one random shape per env), mass re-sampled every reset (0.1-2.0 kg).
The actor only sees proprioception (+ 3-step history) so it must infer the load from how the joints move.
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
from .biped_env_cfg import BipedSceneCfg, BipedWalkEnvCfg

_OBJ_PHYS = dict(
    rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
)
OBJECT_SHAPES = [
    # Squeezed along the robot's lateral axis (object y). Front feet sit at y = +-0.11 m in the hold pose,
    # so lateral widths are 0.18-0.20 m (the feet squeeze slightly inwards).
    sim_utils.CuboidCfg(size=(0.20, 0.20, 0.20), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.55, 0.3)), **_OBJ_PHYS),
    sim_utils.CuboidCfg(size=(0.14, 0.18, 0.14), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.25)), **_OBJ_PHYS),
    sim_utils.CylinderCfg(radius=0.09, height=0.20, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.8)), **_OBJ_PHYS),
    sim_utils.SphereCfg(radius=0.09, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), **_OBJ_PHYS),
]


@configclass
class CarrySceneCfg(BipedSceneCfg):
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=OBJECT_SHAPES,
            random_choice=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)),
    )
    # one filtered sensor per foot (a single multi-body sensor with a filter is not supported by PhysX views)
    fl_foot_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"], history_length=3)
    fr_foot_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"], history_length=3)


@configclass
class CarryEnvCfg(BipedWalkEnvCfg):
    scene: CarrySceneCfg = CarrySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        self.scene.fl_foot_object_contact.update_period = self.sim.dt
        self.scene.fr_foot_object_contact.update_period = self.sim.dt

        # ---- observations: proprioceptive history for the actor, object state for the critic only
        self.observations.policy.history_length = 3
        self.observations.critic.object_pos_b = ObsTerm(func=bmdp.object_pos_b)
        self.observations.critic.object_rel_vel_b = ObsTerm(func=bmdp.object_rel_vel_b)
        self.observations.critic.object_mass = ObsTerm(func=bmdp.object_mass)

        # ---- events: start upright, holding; resample object mass every reset
        self.events.reset_robot = EventTerm(func=bmdp.reset_biped_holding_object, mode="reset")
        self.events.object_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": (0.1, 2.0), "operation": "abs"},
        )
        self.events.push_robot.params = {"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}}
        # adaptive assist curriculum: spring + gravity compensation on the object; every 20 PPO iterations
        # lowered by 0.05 if the mean hold quality is > 0.45, raised if < 0.3 (quality was ~0.5 at assist 0.6).
        self.events.object_assist = EventTerm(
            func=bmdp.adaptive_object_assist, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=True,
            params={"init": 0.6, "step": 0.05, "window": 20 * 24, "lower_if_above": 0.45, "raise_if_below": 0.3},
        )

        # ---- rewards: keep the biped terms, swap 'front feet off ground' for a height-based version
        #      (contact_count would also count touching the object)
        r = self.rewards
        r.front_feet_off_ground = RewTerm(
            func=bmdp.feet_below, weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="F[L,R]_foot"), "height": 0.15},
        )
        r.object_hold = RewTerm(func=bmdp.object_hold_reward, weight=2.0, params={"std": 0.08})
        r.object_grip = RewTerm(func=bmdp.object_grip, weight=1.0,
                                params={"sensor_names": ["fl_foot_object_contact", "fr_foot_object_contact"]})
        r.object_squeeze = RewTerm(func=bmdp.object_squeeze, weight=1.5,
                                   params={"sensor_names": ["fl_foot_object_contact", "fr_foot_object_contact"],
                                           "force_cap": 15.0})
        # wide kernel: with the arms raised (~0.4 m away) a 0.1 m std gave ~zero gradient in carry_v1
        r.hold_pose = RewTerm(func=bmdp.front_feet_hold_pose, weight=2.0, params={"std": 0.3})
        r.object_shake = RewTerm(func=bmdp.object_rel_vel_l2, weight=-0.2)
        r.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["Head_.*", "R.*_hip", "R.*_thigh", "R.*_calf"])

        # ---- termination: dropping the object ends the episode (and costs the termination penalty)
        self.terminations.object_dropped = DoneTerm(func=bmdp.object_dropped,
                                                    params={"max_dist": 0.3, "min_height": 0.15})

        # Commands: start with gentle walking; standing still is common in carrying.
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.3
        cmd.ranges.lin_vel_x = (-0.2, 0.4)
        cmd.ranges.lin_vel_y = (-0.15, 0.15)
        cmd.ranges.ang_vel_z = (-0.6, 0.6)


@configclass
class CarryPlayEnvCfg(CarryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.object_assist = None  # evaluate with the real weight, no assist
