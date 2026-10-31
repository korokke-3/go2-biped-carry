"""HoldV4 (rebuild, 2026-10-22): receive / hold / carry on top of RefWalk-V4.

Changes from HoldRef (agreed with the user 2026-10-21/22):
* objects: boxes and spheres of random size per env (randomize_object_sizes, USD scale before the sim starts):
  boxes 0.10-0.30 deep x 0.08-0.50 wide x 0.06-0.25 high, spheres 0.15-0.36 m; rigid / cardboard / soft each.
  (spheres were 0.15-0.45 m until chunk 4: a 0.44 m sphere placed 1 cm off the belly has its lowest point ~0.28 m in
  front of the base, beyond the ~0.21 m forearms, and rolled off every time; those unholdable spheres kept the level-0
  drop rate at 0.44-0.65 and the curriculum stalled at 0.3 kg. 0.36 m spheres were held, 6-18 % drops.)
  Up to 0.30 m from chunk 8: with the "ha" waiting pose the forearms converge (elbows 18 cm, tips 3 cm apart); a large
  sphere stops against the belly near the tips, on ~6 cm rails, and rolls off sideways - 0.36 m spheres were dropped
  97-100 % in chunks 6 and 7 while the boxes reached 0-8 %.
* curriculum level = lateral size: level 0 only objects >= 0.30 m wide, each level 4 cm narrower (down to 0.08 m);
  then walking (cmd_scale 0 -> 1), then mass beyond 2 kg up to 5 kg (half of the episodes stay <= 2 kg).
* waiting pose = READY_POSE (hips 0.45 rad in, thigh 1.0, calf -1.6; scripts/diagnostics/diag_ready_pose.py): ready_pose, the
  hand-over gate and the reference-state init use it. While holding, the front legs are free (no hip prior).
* fold_arms removed: it rewarded folding the forearms around the object; squeezing is left to the policy (user:
  "no touch").
* speed ranges keep growing with RefWalk-V4's curriculum (commands to object episodes are scaled by cmd_scale).
"""
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

from . import box_mdp as hmdp
from . import refwalk_mdp as rmdp
from .holdref_env_cfg import HoldRefSceneCfg, add_hold_terms
from .refwalk_env_cfg import READY_POSE, RefWalkV4EnvCfg

W_START, W_STEP, W_MIN = 0.30, 0.04, 0.08
V4_MAX_LEVEL = int(round((W_START - W_MIN) / W_STEP)) + 1  # narrowest objects (8 cm) unlock at this level
_BROWN = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.45, 0.3))
_STIFF = [(0.0, 0.0), (5000.0, 100.0), (1000.0, 60.0)]  # rigid / cardboard-like / soft (Hold's compliant contacts)


def _props(k, c):
    return dict(visual_material=_BROWN, activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6,
                                                                compliant_contact_stiffness=k, compliant_contact_damping=c))


@configclass
class HoldV4SceneCfg(HoldRefSceneCfg):
    """HoldRef's sensors and actuators (knee x1.5); the object = 4 unit-size base assets (env % 4), scaled per env."""
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            # 3 boxes + 1 sphere (box_mdp.V4_N_ASSETS / V4_N_BOX). With 3 + 3 the spheres (hard for the converging
            # waiting pose) kept the training drop rate at 0.38-0.40 > 0.35, and the walking stage stalled at
            # cmd_scale 0.3 for two chunks while the boxes were dropped 0-8 %.
            assets_cfg=[sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), **_props(k, c)) for k, c in _STIFF]
            + [sim_utils.SphereCfg(radius=0.5, **_props(*_STIFF[1]))],
            random_choice=False, activate_contact_sensors=True, mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )


@configclass
class HoldV4CurriculumCfg:
    object = CurrTerm(func=hmdp.object_curriculum, params={"start": None})
    cmd_ranges = CurrTerm(func=rmdp.cmd_range_curriculum,
                          params={"lin_term": "tracking_lin_vel", "ang_term": "tracking_ang_vel",
                                  "lin_threshold": 0.8, "ang_threshold": 0.7})


@configclass
class HoldV4EnvCfg(RefWalkV4EnvCfg):
    scene: HoldV4SceneCfg = HoldV4SceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        add_hold_terms(self)
        start = self.curriculum.object.params["start"]
        self.curriculum = HoldV4CurriculumCfg()
        self.curriculum.object.params["start"] = start
        self.curriculum.hind_load = CurrTerm(func=rmdp.load_penalty_curriculum,
                                             params={"term_name": "hind_load", "lin_term": "tracking_lin_vel",
                                                     "step": 0.05, "w_max": 1.0,
                                                     # thresholds for the std^2 0.1 kernel below (RefWalk's 0.8 / 0.7
                                                     # at std^2 0.25 ~ 0.57 / 0.41 here); with 0.8 / 0.7 the weight fell
                                                     # 1.0 -> 0.75 in chunk 3 although the gait did not change
                                                     "up_track": 0.55, "down_track": 0.4})
        self.events.object_sizes = EventTerm(
            func=hmdp.randomize_object_sizes, mode="prestartup",
            params={"box_fwd": (0.10, 0.30), "box_lat": (W_MIN, 0.50), "box_up": (0.06, 0.25), "sphere_d": (0.15, 0.30),
                    "w_start": W_START, "w_step": W_STEP, "max_level": V4_MAX_LEVEL})
        self.events.object_mass.params["p_base"] = 0.5
        self.events.v4_settings = EventTerm(func=v4_settings, mode="startup")
        r = self.rewards
        # holding at zero command drifted forward 0.2 m/s (chunk 2): the ref kernel std^2 0.25 charges that 15 %;
        # std^2 0.1 removed the backward drift of holdref_v4 (holdref_v5)
        r.tracking_lin_vel.params["std"] = 0.1 ** 0.5
        r.fold_arms = None
        r.front_dof_pos_hand.params["target"] = READY_POSE


def v4_settings(env, env_ids):
    """Startup: HoldV4 values for the module-level hand-over state (waiting pose, mass cap)."""
    hmdp.Box.tray_target = list(READY_POSE)
    hmdp.Box.m_cap = 5.0


@configclass
class HoldV4PlayEnvCfg(HoldV4EnvCfg):
    """Evaluation: fixed reference objects (8 old Hold boxes + spheres 0.30 / 0.36 / 0.44 m), see box_mdp.V4_REF_*."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.events.object_sizes.params["fixed_refs"] = True
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.p_base = 0.0


# ----------------------------------------------------------------------------- HoldBall (2026-10-24)
from .refwalk_env_cfg import BALL_JOINTS, BALL_POSE  # noqa: E402

BALL_W_MIN = 0.20
BALL_MAX_LEVEL = int(round((W_START - BALL_W_MIN) / W_STEP)) + 1
BALL_N_ASSETS, BALL_N_BOX = 5, 3
V4_N_ASSETS_REF, V4_N_BOX_REF = hmdp.V4_N_ASSETS, hmdp.V4_N_BOX  # evaluation layout (= HoldV4)


@configclass
class HoldBallSceneCfg(HoldV4SceneCfg):
    """3 boxes (rigid / cardboard / soft) + 2 spheres (cardboard / soft): spheres 2 of 5 (HoldV4: 1 of 4)."""
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), **_props(k, c)) for k, c in _STIFF]
            + [sim_utils.SphereCfg(radius=0.5, **_props(k, c)) for k, c in _STIFF[1:]],
            random_choice=False, activate_contact_sensors=True, mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )


@configclass
class HoldBallEnvCfg(HoldV4EnvCfg):
    """HoldBall: a second holding policy for what HoldV4 (c16, model_31479) cannot hold - large spheres (0.36 / 0.44 m
    dropped 100 %, 0.30 m 6-50 %) - while HoldV4 stays the box specialist (user, 2026-10-24).

    Changes from HoldV4 (everything else, incl. the speed ranges and their curriculum, lateral term, motor-load
    penalty and its curriculum thresholds, joint-limit margin, torsion 0-0.03, lin kernel std^2 0.1, the hand-over,
    the assist / size / mass / walking curriculum, PPO and the Fine switch at assist 0, is HoldV4's):
    * waiting pose BALL_POSE (hips 0.15, thigh 0.5, calf -1.6) instead of READY_POSE (0.45 / 1.0 / -1.6): in the fixed-
      base test the spheres stayed only with an opened "ha" and the upper arms forward (see refwalk_env_cfg.BALL_POSE).
      Walking base: RefWalk-Ball (RefWalk-V4 c5 retrained to hold this pose).
    * training spheres 0.20-0.46 m (HoldV4 0.15-0.30 from chunk 8): 0.36 / 0.44 m were never practised by c16;
      0.46 = the 0.44 m reference + 2 cm. The 0.44 m sphere stalled HoldV4's level 0 only because READY_POSE could not
      keep it at all (fixed base 0 %); BALL_POSE keeps it 0.88 / 0.38.
    * spheres 2 of 5 objects (HoldV4 1 of 4): the target objects get 40 % of the practice instead of 25 %.
    * narrowest object 0.20 m (HoldV4 0.08): BALL_POSE's forearm tips are 22 cm apart and narrower boxes fell through
      in the fixed-base test (0.18 / 0.14 / 0.16 m: 0 / 0 / 0.2 kept); the walking stage starts only after the last
      width level, so unholdable widths would stall it. Narrow boxes are left to HoldV4 (still evaluated here).
    * levels 0-3 (>= 0.30, 0.26, 0.22, 0.20 m) instead of 0-7."""
    scene: HoldBallSceneCfg = HoldBallSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        p = self.events.object_sizes.params
        p.update({"box_lat": (BALL_W_MIN, 0.50), "sphere_d": (BALL_W_MIN, 0.46), "max_level": BALL_MAX_LEVEL,
                  "n_assets": BALL_N_ASSETS, "n_box": BALL_N_BOX,
                  # from chunk 6: half of the spheres 0.38-0.46 m. Chunks 3-4 held 0.30 / 0.36 m (0-25 % drops) but the
                  # 0.44 m one stayed at 69-100 %; uniform 0.20-0.46 gave spheres >= 0.38 only ~12 % of all objects
                  "sphere_big": (0.38, 0.46), "p_sphere_big": 0.5})
        self.events.v4_settings = EventTerm(func=ball_settings, mode="startup")
        self.rewards.front_dof_pos_hand.params["target"] = BALL_POSE
        self.events.reset_robot.params["joint_overrides"] = BALL_JOINTS


def ball_settings(env, env_ids):
    """Startup: HoldBall waiting pose (hand-over gate, reference-state init, ready_pose term) and the mass cap."""
    hmdp.Box.tray_target = list(BALL_POSE)
    hmdp.Box.m_cap = 5.0


@configclass
class HoldBallPlayEnvCfg(HoldBallEnvCfg):
    """Evaluation on HoldV4's fixed reference set with HoldV4's asset layout (3 boxes + 1 sphere, env % 4), so the
    per-object numbers compare 1:1 with HoldV4's."""
    scene: HoldV4SceneCfg = HoldV4SceneCfg(num_envs=16, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.events.object_sizes.params.update({"fixed_refs": True, "n_assets": V4_N_ASSETS_REF, "n_box": V4_N_BOX_REF})
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.p_base = 0.0



@configclass
class HoldBallRobustEnvCfg(HoldBallEnvCfg):
    """HoldBall + motor-strength randomization (refwalk_mdp.randomize_motor_strength, per env x0.8-1.5 on the torque-
    speed curve). HoldBall c6 (17988) fell in MuJoCo with flat motor limits while standing up and did not fall with
    the training curve or with a flat 30 Nm limit; 1.5 = MuJoCo's / the real knee's 45.4 Nm over the curve's 30.3 Nm,
    0.8 = the -20 % planned for weaker real motors (CLAUDE.md, 10-24)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.motor_strength = EventTerm(func=rmdp.randomize_motor_strength, mode="reset",
                                               params={"scale_range": (0.8, 1.5)})
