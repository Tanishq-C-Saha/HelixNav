
import gymnasium as gym
from . import agents


# ── CP7 (PBRS reward, curriculum randomisation) ──
gym.register(
    id="HelixNav-CP7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "helix_nav.tasks.manager_based.navigation.config.go2"
            ".env_configs.cp7.cp7_env:HelixNavCP7RLEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="HelixNav-CP7-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "helix_nav.tasks.manager_based.navigation.config.go2"
            ".env_configs.cp7.cp7_env:HelixNavCP7RLEnvPlayCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)