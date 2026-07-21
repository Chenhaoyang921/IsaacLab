# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""事件函式：startup 掛末端負載；reset 重置機器人與任務狀態。"""

from __future__ import annotations


def apply_payload(env, env_ids=None):
    """startup：把末端模擬物品的質量/慣量加到 panda_hand（只執行一次）。"""
    env.power_task.apply_payload()


def reset_power_task(env, env_ids=None):
    """reset：重置機器人狀態與任務緩衝（路徑點索引、計時器等）。"""
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = None
    env.power_task.reset(env_ids)
