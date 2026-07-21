# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka 最小功率走點任務（manager-based workflow）。"""

import gymnasium as gym

from . import agents

gym.register(
    id="Franka-Power-Opt-Mgr-v0",
    entry_point=f"{__name__}.franka_power_env:FrankaPowerManagerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_power_env_cfg:FrankaPowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPowerMgrPPORunnerCfg",
    },
)
