from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class BipedPPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 3000
    save_interval = 200


@configclass
class BipedSymPPORunnerCfg(BipedPPORunnerCfg):
    """Left-right mirror data augmentation (Mittal et al. 2024; Isaac Lab symmetry/anymal.py pattern)."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

        from .symmetry import compute_symmetric_states

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=compute_symmetric_states
        )


@configclass
class BipedSymLowEntPPORunnerCfg(BipedSymPPORunnerCfg):
    """Fine-tuning variant: entropy 0.01 -> 0.0025 (Isaac Lab's Spot config). Load-V2 (0.01) let the action std
    climb 0.61 -> 1.33 and collapsed from ~5 % to ~66 % falls after 700 iterations."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0025


@configclass
class BipedSymLossPPORunnerCfg(BipedSymLowEntPPORunnerCfg):
    """Augmentation + mirror loss (rsl_rl use_mirror_loss; Mittal et al. 2024 'PPOloss'). Load-V3 trained with
    augmentation only drifted from a symmetric warm start (mean |pi(mirror o) - mirror(pi o)| 0.02 -> 0.60):
    stand still with one foot 1 rad ahead, right turns with a 0.6 rad hip-pitch offset that left turns lack."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg.use_mirror_loss = True
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 1.0


@configclass
class BipedSymLossFinePPORunnerCfg(BipedSymLossPPORunnerCfg):
    """Fine-tuning holding / carrying: entropy 0.0025 -> 0.001. In hold_v8 and carry_walk_v1 the action std crept up
    0.08 -> 0.17 while the drop rate rose 17 % -> 29 % (sampled-action noise makes the arms shake the object: 5-7 %
    drops with the mean action vs 24-28 % sampled, diag_hold_counter.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.001
