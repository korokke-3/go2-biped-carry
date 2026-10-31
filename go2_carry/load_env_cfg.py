"""Stage 3b: biped stand/walk with an UNKNOWN payload suddenly placed on / removed from the front forearms.

The payload is a world-frame downward force on the two front calves (0.1-2.0 kg total, +-30% left/right
asymmetry, 30% of the time no load), re-sampled every 2-6 s. The actor sees only proprioception with a
3-step history (it has to *feel* the load); the critic additionally sees the true loads.
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp as bmdp
from .biped_env_cfg import BipedGaitWalkV6bEnvCfg, BipedWalkEnvCfg


@configclass
class LoadEnvCfg(BipedWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 3
        self.observations.critic.forearm_load = ObsTerm(func=bmdp.forearm_load_obs)
        self.events.forearm_load = EventTerm(
            func=bmdp.forearm_load, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=True,
            params={"mass_range": (0.1, 2.0), "hold_range_s": (2.0, 6.0), "p_none": 0.3, "asym": 0.3},
        )
        # keep the CoM (incl. payload) over the hind feet: helps both payload balance and the stand-still drift
        self.rewards.com_over_feet = RewTerm(func=bmdp.loaded_com_over_hind_feet, weight=1.0, params={"std": 0.1})
        # front legs: arms must not hang onto the ground under load
        self.rewards.front_feet_off_ground.weight = -1.0
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.3
        cmd.ranges.lin_vel_x = (-0.3, 0.4)
        cmd.ranges.lin_vel_y = (-0.15, 0.15)
        cmd.ranges.ang_vel_z = (-0.6, 0.6)


@configclass
class LoadPlayEnvCfg(LoadEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class LoadV2EnvCfg(BipedGaitWalkV6bEnvCfg):
    """Load task rebuilt on GaitWalk v6b (gait clock, symmetric training, heading tracking; passed the walking
    criteria on 2026-10-17); same payload model and observation additions as LoadEnvCfg. Warm-started from v6b
    model_15999 with scripts/warmstart.py (actor 3,3,3,12,12,12,2,1 with history 3; critic 66 -> 68)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 3
        self.observations.critic.forearm_load = ObsTerm(func=bmdp.forearm_load_obs)
        self.events.forearm_load = EventTerm(
            func=bmdp.forearm_load, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=True,
            params={"mass_range": (0.1, 2.0), "hold_range_s": (2.0, 6.0), "p_none": 0.3, "asym": 0.3},
        )
        self.rewards.com_over_feet.weight = 1.0
        self.rewards.front_feet_off_ground.weight = -1.0
        # payload makes walking harder: restart at half of v6b's final speed range, let the curriculum regrow it
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.3, 0.5), (-0.2, 0.2), (-1.2, 1.2)
        self.commands.base_velocity.rel_standing_envs = 0.3


@configclass
class LoadV2PlayEnvCfg(LoadV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class LoadV5EnvCfg(LoadV2EnvCfg):
    """Load-V5: CoM-over-feet weight back to v6b's 0.5. Lateral walking (0.2 m/s) reached 79-87 % in v6b
    (weight 0.5) but only 58-69 % (Load-V3) and 26-34 % (Load-V4) with weight 1.0: side steps move the CoM
    off the feet midpoint. Everything else as Load-V4 (mirror loss, entropy 0.0025)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.com_over_feet.weight = 0.5


@configclass
class LoadV5PlayEnvCfg(LoadV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class LoadV6EnvCfg(LoadV5EnvCfg):
    """Load-V6: + lateral-only tracking term (std^2 0.02 = the xy kernel's 0.1 scaled by (0.4 / 1.0)^2, the ratio
    of the lateral to the forward command range). Load-V5 side-stepped at 32-41 % of the command."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_y = RewTerm(func=bmdp.track_biped_lin_vel_y_exp, weight=0.5,
                                               params={"command_name": "base_velocity", "std": 0.02 ** 0.5})


@configclass
class LoadV6PlayEnvCfg(LoadV6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class LoadV7EnvCfg(LoadV6EnvCfg):
    """Load-V7: lateral tracking weight 0.5 -> 1.0. In Load-V6 the lateral term plateaued at 74 % of its weight
    after ~500 iterations (steady side-step speed 62-75 % of the command): an equilibrium with the effort
    penalties, so more iterations alone do not help."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_y.weight = 1.0


@configclass
class LoadV7PlayEnvCfg(LoadV7EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
