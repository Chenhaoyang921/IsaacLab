# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""階段感知自訂動作類別。

StageGatedJointPositionAction   — 手臂：第二階段 [0,1,0,0] 輸出零偏移，凍結姿態
ProgrammaticGripperAction       — 夾爪：由任務階段決定開閉，RL 不佔動作維度
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionAction, BinaryJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, BinaryJointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ──────────────────────────────────────────────────────────────────────────────
# 手臂動作：第二階段凍結
# ──────────────────────────────────────────────────────────────────────────────

class StageGatedJointPositionAction(JointPositionAction):
    """關節位置動作，第二階段 [0,1,0,0] 時輸出零偏移（凍結手臂姿態）。"""

    def process_actions(self, actions: torch.Tensor) -> None:
        from .rewards import _compute_stage_tag

        stage_tag = _compute_stage_tag(self._env)
        in_stage2 = stage_tag[:, 1].bool()       # [0,1,0,0] = 第二階段

        modified = actions.clone()
        modified[in_stage2] = 0.0                # 零偏移 → 保持當前位置
        super().process_actions(modified)


@configclass
class StageGatedJointPositionActionCfg(JointPositionActionCfg):
    """StageGatedJointPositionAction 的配置類別。"""
    class_type: type = StageGatedJointPositionAction


# ──────────────────────────────────────────────────────────────────────────────
# 夾爪動作：由階段程式控制，RL 不佔動作維度
# ──────────────────────────────────────────────────────────────────────────────

class ProgrammaticGripperAction(BinaryJointPositionAction):
    """夾爪由任務階段控制，RL 策略不參與。

    第一階段 [1,0,0,0]：夾爪張開（接近花朵）
    其餘所有階段     ：夾爪閉合（抓取 / 插入）
    """

    @property
    def action_dim(self) -> int:
        return 0  # 不佔用 RL 動作向量的任何維度

    def process_actions(self, actions: torch.Tensor) -> None:
        """忽略 RL 輸入（actions 為空張量），改由階段決定開閉訊號。"""
        from .rewards import _compute_stage_tag

        stage_tag = _compute_stage_tag(self._env)
        # 第一階段 → 1.0（張開），其餘 → 0.0（閉合）
        open_signal = stage_tag[:, 0:1]          # (num_envs, 1)

        # 呼叫父類別，以 open_signal 作為二元訊號（>0 = 開，≤0 = 關）
        super().process_actions(open_signal)


@configclass
class ProgrammaticGripperActionCfg(BinaryJointPositionActionCfg):
    """ProgrammaticGripperAction 的配置類別。"""
    class_type: type = ProgrammaticGripperAction


# ──────────────────────────────────────────────────────────────────────────────
# 夾爪動作：帶 1 秒冷卻的二元控制
# ──────────────────────────────────────────────────────────────────────────────

class CooldownBinaryGripperAction(BinaryJointPositionAction):
    """二元夾爪動作，每次開/關切換後鎖定 0.1 秒才能再次改變。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._cooldown_steps = int(1.0 / env.step_dt)
        self._last_command = torch.ones(env.num_envs, 1, device=env.device)
        self._cooldown_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._gripper_locked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        num_steps_per_env = 192
        current_epoch = self._env.common_step_counter // num_steps_per_env

        # 環境重置時清除所有狀態
        reset_mask = self._env.episode_length_buf == 0
        self._cooldown_counter[reset_mask] = 0
        self._gripper_locked[reset_mask] = False

        # epoch < 100：無冷卻，直接透傳
        if current_epoch < 100:
            self._last_command = actions.clone()
            super().process_actions(actions)
            return

        effective = actions.clone()

        # epoch >= 200：閉合後永久鎖定，不能再張開
        if current_epoch >= 200:
            effective[self._gripper_locked] = -1.0
            just_closed = (~(effective > 0).squeeze(-1)) & ~self._gripper_locked
            self._gripper_locked[just_closed] = True
            self._last_command = effective.clone()
            super().process_actions(effective)
            return

        # epoch 100~199：1 秒冷卻
        on_cooldown = self._cooldown_counter > 0
        effective[on_cooldown] = self._last_command[on_cooldown]

        new_sign = (effective > 0).float()
        old_sign = (self._last_command > 0).float()
        changed = (new_sign != old_sign).squeeze(-1)

        self._cooldown_counter[changed] = self._cooldown_steps
        self._cooldown_counter = (self._cooldown_counter - 1).clamp(min=0)

        self._last_command = effective.clone()
        super().process_actions(effective)


@configclass
class CooldownBinaryGripperActionCfg(BinaryJointPositionActionCfg):
    """CooldownBinaryGripperAction 的配置類別。"""
    class_type: type = CooldownBinaryGripperAction
