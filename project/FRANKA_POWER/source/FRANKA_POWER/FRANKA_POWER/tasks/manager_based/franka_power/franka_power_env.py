# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka 最小功率走點任務（manager-based workflow）。

與 Direct 版功能等價，差別在於觀測 / 獎勵 / 終止 / 動作 / 事件都拆成
manager 的宣告式 term（見 franka_power_env_cfg.py 與 mdp/），共用狀態集中在
``env.power_task``（PowerTaskEngine）。

這個薄薄的 ManagerBasedRLEnv 子類別唯一的職責，是在 episode 重置時把
「各關節平均力矩上限」「成功率」「平均功率」等自訂 metric 補記到 log —
這些是專案的核心產出，而 RewardManager 已自動記錄 ``Episode_Reward/*``。
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv


class FrankaPowerManagerEnv(ManagerBasedRLEnv):
    def _reset_idx(self, env_ids):
        # 在 super() 重置任務緩衝「之前」，先擷取剛結束 episode 的自訂 metric
        metrics = self.power_task.episode_metrics(env_ids)
        # super() 會清空並重建 extras["log"]，同時自動填入 Episode_Reward/* 等
        super()._reset_idx(env_ids)
        # 補上專案自訂 metric（EffortLimit_Nm/*、success_rate、mean_power_W ...）
        self.extras["log"].update(metrics)
