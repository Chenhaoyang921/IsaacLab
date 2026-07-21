# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""自訂 ActionTerm：把 7 維動作解讀成各關節力矩上限，並用差分 IK 追蹤路徑點。

所有實際運算都委派給 ``PowerTaskEngine``；這個 term 只負責：
* 在建構時建立引擎，並掛到 ``env.power_task`` 供其他 mdp 函式取用。
* 把 ActionManager 的每步呼叫（process/apply）轉接到引擎。
"""

from __future__ import annotations

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from ..power_task import PowerTaskEngine


class EffortLimitIKAction(ActionTerm):
    """動作 = 7 個關節的力矩上限（正規化 [-1,1]）；軌跡由差分 IK 產生。"""

    cfg: EffortLimitIKActionCfg

    def __init__(self, cfg: EffortLimitIKActionCfg, env):
        super().__init__(cfg, env)
        self.engine = PowerTaskEngine(env, asset_name=cfg.asset_name)
        # 掛到 env 供 observation / reward / termination / 事件取用
        env.power_task = self.engine
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.engine.effort_limits

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self.engine.begin_step(actions)

    def apply_actions(self):
        self.engine.apply()

    def reset(self, env_ids=None):
        # 任務緩衝的重置由 reset 事件（events.reset_power_task）統一處理
        pass


@configclass
class EffortLimitIKActionCfg(ActionTermCfg):
    class_type: type = EffortLimitIKAction
    asset_name: str = "robot"
