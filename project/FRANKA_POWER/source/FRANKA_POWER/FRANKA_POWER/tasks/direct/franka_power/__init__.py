"""Franka 最小功率走點任務。"""

import gymnasium as gym

from . import agents

##
# 註冊 Gym 環境
##

gym.register(
    id="Franka-Power-Opt-v0",
    entry_point=f"{__name__}.franka_power_env:FrankaPowerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_power_env_cfg:FrankaPowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:FrankaPowerPPORunnerCfg",
    },
)
