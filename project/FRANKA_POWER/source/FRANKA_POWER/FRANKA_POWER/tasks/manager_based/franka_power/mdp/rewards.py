# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""獎勵函式：只回傳「原始量」，實際權重放在 RewardsCfg 的 RewTerm.weight。

RewardManager 會把每個 term 算成 ``func * weight * dt``，並自動累加成
``Episode_Reward/<term_name>`` 記到 log。dt 是所有 term 共同的常數因子，
因此各項之間的相對平衡與 Direct 版完全一致（只差一個全域尺度 dt）。

每個函式開頭呼叫 ``engine.update()``（每步只真正計算一次）以確保狀態最新。
"""

from __future__ import annotations

import torch


def progress(env) -> torch.Tensor:
    """朝當前目標點前進的距離（公尺）。"""
    env.power_task.update()
    return env.power_task.progress


def waypoint_bonus(env) -> torch.Tensor:
    """這一步是否到達了一個路徑點。"""
    env.power_task.update()
    return env.power_task.reached.float()


def success_bonus(env) -> torch.Tensor:
    """這一步是否走完全部路徑點。"""
    env.power_task.update()
    return env.power_task.task_done.float()


def speed_bonus(env) -> torch.Tensor:
    """走完全程那一刻的剩餘時間比例（越快越大，未完成為 0）。"""
    env.power_task.update()
    return env.power_task.speed_frac


def distance_penalty(env) -> torch.Tensor:
    """離當前目標點的距離（持續懲罰）。"""
    env.power_task.update()
    return env.power_task.dist


def energy_penalty(env) -> torch.Tensor:
    """這一步消耗的電能（焦耳）。"""
    env.power_task.update()
    return env.power_task.energy_J


def effort_limit_penalty(env) -> torch.Tensor:
    """平均力矩上限比例（懲罰開太大的功率）。"""
    env.power_task.update()
    return env.power_task.limit_norm


def action_rate_penalty(env) -> torch.Tensor:
    """動作變化率平方和（讓功率設定平滑）。"""
    env.power_task.update()
    return env.power_task.action_rate


def timeout_penalty(env) -> torch.Tensor:
    """是否超過路徑點時間預算。"""
    env.power_task.update()
    return env.power_task.timeout.float()


def failure_penalty(env) -> torch.Tensor:
    """脫離軌跡失敗（不含超時，避免同一步被扣兩次）。"""
    env.power_task.update()
    e = env.power_task
    return (e.task_failed & ~e.timeout).float()
