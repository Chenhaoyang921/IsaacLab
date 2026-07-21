# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""終止條件函式。時間上限（truncation）用內建的 mdp.time_out。"""

from __future__ import annotations

import torch


def task_success(env) -> torch.Tensor:
    """走完全部路徑點 -> 成功終止。"""
    env.power_task.update()
    return env.power_task.task_done


def task_failure(env) -> torch.Tensor:
    """脫離軌跡或超時 -> 失敗終止。"""
    env.power_task.update()
    return env.power_task.task_failed
