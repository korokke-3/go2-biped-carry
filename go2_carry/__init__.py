"""Go2 bipedal carrying tasks (Isaac Lab). Task ids contain "Unitree" so unitree_rl_lab's scripts accept them."""
import gymnasium as gym


def _patch_rsl_rl_resume_lr():
    """rsl_rl's OnPolicyRunner.load() restores the optimizer state, but PPO's adaptive schedule keeps its own
    ``learning_rate`` attribute at the config value (1e-3) and writes it back into the optimizer at the first update.
    Resumed runs therefore jumped from ~3e-5..2e-4 to 1e-3 and regressed for hundreds of iterations (hold_v8:
    drop rate 16 % -> 30 %, carry_walk_v1/v2). Sync the attribute from the loaded optimizer."""
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        return
    if getattr(OnPolicyRunner.load, "_lr_synced", False):
        return
    _orig = OnPolicyRunner.load

    def load(self, path, load_optimizer=True, *a, **k):
        out = _orig(self, path, load_optimizer, *a, **k)
        if load_optimizer and hasattr(self.alg, "optimizer") and hasattr(self.alg, "learning_rate"):
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
            print(f"[go2_carry] resumed learning rate {self.alg.learning_rate:.3g}", flush=True)
        return out

    load._lr_synced = True
    OnPolicyRunner.load = load


_patch_rsl_rl_resume_lr()

gym.register(
    id="Unitree-Go2-Biped-Stand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedStandEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedStandPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedWalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedWalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Carry",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carry_env_cfg:CarryEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.carry_env_cfg:CarryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Load",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-GaitWalk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-GaitWalk-V2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-GaitWalk-V3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV3EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-GaitWalk-V4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV4EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV4PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-GaitWalk-V5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV5EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV5PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Load-V2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-GaitWalk-V6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV6EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV6PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-GaitWalk-V6b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV6bEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedGaitWalkV6bPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Load-V3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLowEntPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Load-V4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Load-V5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV5EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV5PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Load-V6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV6EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV6PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Load-V7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV7EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.load_env_cfg:LoadV7PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Box",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.box_env_cfg:BoxEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.box_env_cfg:BoxPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Hold",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hold_env_cfg:HoldEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.hold_env_cfg:HoldPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-Carry-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hold_env_cfg:HoldEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.hold_env_cfg:HoldPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossFinePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-RefWalk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldRef",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdref_env_cfg:HoldRefEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdref_env_cfg:HoldRefPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-RefWalk-V2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldRef-Fine",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdref_env_cfg:HoldRefEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdref_env_cfg:HoldRefPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossFinePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-RefWalk-V3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkV3EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkV3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldRef-V2-Fine",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdref_env_cfg:HoldRefV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdref_env_cfg:HoldRefV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossFinePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldRef-V3-Fine",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdref_env_cfg:HoldRefV3EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdref_env_cfg:HoldRefV3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossFinePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-RefWalk-V4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkV4EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkV4PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldV4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldV4EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldV4PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldV4-Fine",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldV4EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldV4PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossFinePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-RefWalk-Ball",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkBallEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.refwalk_env_cfg:RefWalkBallPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldBall",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldBallEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldBallPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldBall-Fine",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldBallEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldBallPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossFinePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Biped-HoldBall-Robust-Fine",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldBallRobustEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.holdv4_env_cfg:HoldBallPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedSymLossFinePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Tripod",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-TripodHigh",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodHighEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodHighPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-TripodCool",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodCoolEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodCoolPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-TripodHind",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodHindEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodHindPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)

# biped <-> quadruped switching + measured real joints (switch_env_cfg.py, 2026-10-31): the first chunk with entropy
# 0.0025 (BipedSymLossPPORunnerCfg) to explore the new transitions, then -Fine (0.001) as the holding lines
for _name, _cls in (("HoldBall", "HoldBallSwitch"), ("HoldV4", "HoldV4Switch")):
    for _suffix, _agent in (("", "BipedSymLossPPORunnerCfg"), ("-Fine", "BipedSymLossFinePPORunnerCfg")):
        gym.register(
            id=f"Unitree-Go2-Biped-{_name}-Switch{_suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.switch_env_cfg:{_cls}EnvCfg",
                "play_env_cfg_entry_point": f"{__name__}.switch_env_cfg:{_cls}PlayEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents:{_agent}",
            },
        )

gym.register(
    id="Unitree-Go2-TripodReal",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodRealEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tripod_env_cfg:TripodRealPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BipedPPORunnerCfg",
    },
)
