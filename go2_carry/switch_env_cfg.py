"""HoldBall-Switch / HoldV4-Switch (2026-10-31): the two holding policies + biped <-> quadruped switching + the
measured real-robot joints (user, 2026-10-31: the transitions first, for both holding policies).

go2_quad2hand (ref) vs here, item by item:
  gait-mode one-hot in the observation [quad, biped] ........... same (appended after the clock; warm start with zero
                                                                  input weights, scripts/warmstart.py)
  mode re-drawn every 10 s, uniformly .......................... flips every U(3, 8) s, only while no object is held
                                                                  (user: go down with empty hands); 3 s covers the
                                                                  stand-up (~1.5 s), 8 s < the 20 s episode, so most
                                                                  object-free episodes see both directions
  mode-specific rewards x (mode == m) .......................... same: all biped terms gated biped; quadruped terms as
                                                                  the ref's quad mode / Tripod at zero command
    tracking 1.5 / 1.0 (sigma^2 0.25) .......................... same, on the zero command (stand still)
    base_height_quad 0.36 (sigma 0.04) 1.0 ..................... same
    body_orientation_quad pitch 0 (sigma 0.04) 1.0 ............. same + a wide copy (sigma 1.0, 0.5): from upright
                                                                  (pitch 90 deg) the narrow term is e^-62, flat ground for
                                                                  the transition; the wide one pays 0.08 upright -> 1 flat
    gait_quad 0.5 (always stepping) ............................ fraction of the four feet down, 0.5: the rest is
                                                                  standing still (Tripod: no stepping at zero command)
    hip_close_to_default -1.0 .................................. as before (hind hips) + all joints to the default
                                                                  pose -0.1 in the quad mode (the controller lies down
                                                                  from there for cooling)
  no transition reward / no transition-specific termination ..... same
Episodes: objects in 60 % (HoldV4 / HoldBall 80 %): object-free time is where the switching is learnt.
Curricula: the command ranges and the motor-load weight were at their limits in both lines (x -0.8..1.2, y +-0.5,
z +-1.5, weight -1.0; runs/holdball.*, runs/holdv4.*): fixed there. Their curricula read the gated tracking terms and
would misread the quadruped time as poor tracking (cf. the 10-23 load-curriculum episode-length error). The object
curriculum (drop rate) continues.

Real joints (switch_mdp.REAL_JOINTS, measured on a real Go2): Fs / mu / Fd = measured x U(0.5, 1.5) per env and
joint type; knee rotor inertia 0.03-0.08 (around the measured value; hips / thighs keep 0.01-0.05); base mass
-1..+2 kg; motor strength x0.8-1.5 (HoldBall-Robust). Quadruped resets spread the joints
+-0.15 rad (training: +-0.05; the real robot ran from 0.14 and tipped over from 0.30, deploy/README.md).
"""
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import refwalk_mdp as rmdp
from . import switch_mdp as smdp
from . import tripod_mdp as tmdp
from .holdv4_env_cfg import HoldBallRobustEnvCfg, HoldV4EnvCfg, HoldV4SceneCfg
from .refwalk_env_cfg import ALL_FEET

BIPED_TERMS = ["tracking_lin_vel", "tracking_ang_vel", "tracking_lin_vel_y", "gait_hand", "base_height_hand",
               "body_orientation_hand", "foot_clearance_hand", "front_contact_hand", "front_dof_pos_hand", "ready_pose",
               "com_over_feet"]
SWITCH_S = (3.0, 8.0)
# one-hot x5 (10-31): the warm-started runs resume at lr 1.5e-5..5e-5 (adaptive, KL-limited); Adam moves each weight by
# ~lr per step, so the new input's effect grows with its size. At x1 the mode input stayed unused for 2100
# iterations (the policy stood up in the quadruped mode although holding the stance paid 6.3 / s vs 2.6 / s,
# scripts/diagnostics/diag_switch_reward.py). scripts/export_onnx.py --switch writes the same scale.
MODE_OBS_SCALE = 5.0


def add_switch_terms(cfg, p_object: float = 0.6):
    o = cfg.observations
    o.policy.gait_mode = ObsTerm(func=smdp.gait_mode_onehot, scale=MODE_OBS_SCALE)
    o.critic.gait_mode = ObsTerm(func=smdp.gait_mode_onehot, scale=MODE_OBS_SCALE)
    cfg.events.object_handover.params["p_object"] = p_object
    # quad_up_max 0.99: the mid resets (pitch 0-81 deg) may start in the quadruped mode too (reference-state init,
    # DeepMimic, Peng et al. 2018): chunk 1 with quadruped starts only (< 0.3) never went down from upright (10-31,
    # quad_flat 0.0014 for 1500 iterations, and the policy stood up in the quadruped mode); from the mid poses the way
    # down is short and the start pitch spreads it to the upright pose
    cfg.events.mode_switch = EventTerm(func=smdp.mode_switch, mode="interval", interval_range_s=(0.02, 0.02),
                                       is_global_time=True, params={"switch_s": SWITCH_S, "p_quad_start": 0.5,
                                                                    "quad_up_max": 0.99})
    cfg.events.reset_robot.params["quad_joint_noise"] = 0.15
    if os.environ.get("SWITCH_OFF"):
        # (10-31) always biped: the transitions were not learnt (2600 iterations, 4 variants, logs/summary_switch.txt),
        # the robot goes down by a scripted move (deploy/go2_real.py --down scripted); these runs keep the measured
        # real joints and the observation layout (mode one-hot fixed at biped)
        cfg.events.mode_switch.params.update({"fixed_mode": smdp.BIPED, "fixed_switch_s": 1e9})
    r = cfg.rewards
    for name in BIPED_TERMS:
        t = getattr(r, name, None)
        if t is not None:
            t.func = smdp.gated(t.func, smdp.BIPED)
    q = smdp.QUAD
    r.quad_still_lin = RewTerm(func=smdp.gated(mdp.track_lin_vel_xy_yaw_frame_exp, q), weight=1.5,
                               params={"command_name": "base_velocity", "std": 0.5})
    r.quad_still_yaw = RewTerm(func=smdp.gated(mdp.track_ang_vel_z_world_exp, q), weight=1.0,
                               params={"command_name": "base_velocity", "std": 0.5})
    r.quad_height = RewTerm(func=smdp.gated(rmdp.base_height_hand, q), weight=1.0, params={"target": 0.36, "sigma": 0.04})
    r.quad_flat = RewTerm(func=smdp.gated(tmdp.base_orientation_flat, q), weight=1.0, params={"sigma": 0.04})
    r.quad_flat_wide = RewTerm(func=smdp.gated(tmdp.base_orientation_flat, q), weight=0.5, params={"sigma": 1.0})
    # linear in cos(pitch): the exp terms are flat at the upright pose (gradient of the wide one 0.03 / rad there)
    r.quad_level = RewTerm(func=smdp.gated(smdp.body_level, q), weight=1.0)
    r.quad_feet = RewTerm(func=smdp.gated(smdp.feet_contact_frac, q), weight=0.5,
                          params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ALL_FEET, preserve_order=True)})
    r.quad_pose = RewTerm(func=smdp.gated(rmdp.joint_dev_l2, q), weight=-0.1,
                          params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    # base / head on the ground: a fall (termination -200) only in the biped mode; in the quadruped mode a penalty
    # (10-31: chunk 1 and two probes never tried to go down; to a policy trained not to fall for 25000 iterations
    # any lean towards the ground risked -200 against +3.7 / s for the quadruped stance; a belly landing still pays
    # ~2 / s less than the stance)
    cfg.terminations.body_contact.func = smdp.biped_only(cfg.terminations.body_contact.func)
    r.quad_body_contact = RewTerm(func=smdp.gated(cfg.terminations.body_contact.func.__wrapped__, q), weight=-2.0,
                                  params=dict(cfg.terminations.body_contact.params))
    # fixed command ranges / load weight (see the module docstring)
    c = cfg.commands.base_velocity
    c.ranges = c.limit_ranges
    cfg.curriculum.cmd_ranges = None
    cfg.curriculum.hind_load = None
    r.hind_load.weight = -1.0


def add_real_joint_terms(cfg):
    e = cfg.events
    e.joint_friction_real = EventTerm(func=smdp.randomize_joint_friction_real, mode="reset",
                                      params={"scale_range": (0.5, 1.5)})
    e.knee_armature = EventTerm(func=mdp.randomize_joint_parameters, mode="startup",
                                params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
                                        "armature_distribution_params": (0.03, 0.08), "operation": "abs"})
    e.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
    if getattr(e, "motor_strength", None) is None:
        e.motor_strength = EventTerm(func=rmdp.randomize_motor_strength, mode="reset", params={"scale_range": (0.8, 1.5)})


def play_switch(cfg):
    """Evaluation: measured joints at their nominal values; SWITCH_MODE=quad|biped fixes the start mode and
    SWITCH_AT=<s> flips object-free envs once at that time (scripts/evaluate.py sets them)."""
    cfg.events.joint_friction_real.params["scale_range"] = (1.0, 1.0)
    cfg.events.motor_strength = None
    p = cfg.events.mode_switch.params
    if os.environ.get("SWITCH_MODE"):
        p["fixed_mode"] = {"quad": smdp.QUAD, "biped": smdp.BIPED}[os.environ["SWITCH_MODE"]]
    if os.environ.get("SWITCH_AT"):
        p["fixed_switch_s"] = float(os.environ["SWITCH_AT"])
    cfg.events.reset_robot.params["quad_joint_noise"] = 0.05


@configclass
class HoldBallSwitchEnvCfg(HoldBallRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_switch_terms(self)
        add_real_joint_terms(self)


@configclass
class HoldBallSwitchPlayEnvCfg(HoldBallSwitchEnvCfg):
    """holdv4_env_cfg.HoldBallPlayEnvCfg's evaluation set-up (HoldV4's fixed reference objects and asset layout)."""
    scene: HoldV4SceneCfg = HoldV4SceneCfg(num_envs=16, env_spacing=2.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        _ball_play(self)
        play_switch(self)


def _ball_play(cfg):
    from .holdv4_env_cfg import V4_N_ASSETS_REF, V4_N_BOX_REF
    cfg.scene.num_envs = 16
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.curriculum = None
    cfg.events.object_sizes.params.update({"fixed_refs": True, "n_assets": V4_N_ASSETS_REF, "n_box": V4_N_BOX_REF})
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges
    cfg.commands.base_velocity.p_base = 0.0


@configclass
class HoldV4SwitchEnvCfg(HoldV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_switch_terms(self)
        add_real_joint_terms(self)


@configclass
class HoldV4SwitchPlayEnvCfg(HoldV4SwitchEnvCfg):
    """holdv4_env_cfg.HoldV4PlayEnvCfg's evaluation set-up (fixed reference objects)."""

    def __post_init__(self):
        super().__post_init__()
        _v4_play(self)
        play_switch(self)


def _v4_play(cfg):
    cfg.scene.num_envs = 16
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.curriculum = None
    cfg.events.object_sizes.params["fixed_refs"] = True
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges
    cfg.commands.base_velocity.p_base = 0.0
