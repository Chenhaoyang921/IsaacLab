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
        from .rewards import _update_stage_state

        _update_stage_state(self._env)  # 強制刷新階段狀態，確保讀取到最新標籤
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
# 夾爪動作：抓到花之後鎖定
# ──────────────────────────────────────────────────────────────────────────────

class CooldownBinaryGripperAction(BinaryJointPositionAction):
    """二元夾爪動作：抓到花（is_catch）之後鎖定，不再接受 RL 輸入。

    epoch < LOCK_AFTER_EPOCH 期間不啟用鎖定，全程由 RL 自由控制，讓前期能自由探索抓取。
    鎖定的指令是「首次抓到當下生效的那個指令」。因為 is_catch 需要手指施力 >= 1N，
    成立時該指令必然是閉合，所以等於一路握到回合結束。回合重置時解鎖。
    """

    # 從第幾個 epoch 開始啟用鎖定
    LOCK_AFTER_EPOCH = 30
    # 每個 epoch 的步數，需與 rsl_rl_ppo_cfg.py 的 num_steps_per_env 一致
    NUM_STEPS_PER_EPOCH = 192

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._last_command = torch.ones(env.num_envs, 1, device=env.device)
        self._locked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._latched_command = torch.zeros(env.num_envs, 1, device=env.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        from .rewards import is_catch

        # epoch 30 之前：不鎖定，直接透傳
        current_epoch = self._env.common_step_counter // self.NUM_STEPS_PER_EPOCH
        if current_epoch < self.LOCK_AFTER_EPOCH:
            self._locked[:] = False
            self._last_command = actions.clone()
            super().process_actions(actions)
            return

        # 新回合解鎖（process_actions 在該回合第一步時 episode_length_buf 尚為 0）
        reset_mask = self._env.episode_length_buf == 0
        self._locked[reset_mask] = False

        # 已鎖定的環境忽略 RL 輸入，沿用鎖定當下的指令
        effective = actions.clone()
        effective[self._locked] = self._latched_command[self._locked]

        # 首次抓到 → 記下當下指令並上鎖
        caught = is_catch(self._env).bool()
        newly_locked = caught & ~self._locked
        self._latched_command[newly_locked] = effective[newly_locked]
        self._locked |= caught

        self._last_command = effective.clone()
        super().process_actions(effective)


@configclass
class CooldownBinaryGripperActionCfg(BinaryJointPositionActionCfg):
    """CooldownBinaryGripperAction 的配置類別。"""
    class_type: type = CooldownBinaryGripperAction
