# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""觀測函式。

全部讀「即時狀態」（直接從機器人資料 + 引擎的 wp_idx / 動作緩衝計算），
不經過 ``engine.update()`` 的 post-physics 快取——這樣 reset 之後回傳的觀測
才會反映新 episode 的狀態，而不是上一個 episode 的殘值。
"""

from __future__ import annotations

import torch


def robot_state(env) -> torch.Tensor:
    """手臂關節相對角(7) + 縮放速度(7) + 當前功率上限正規化動作(7) = 21。"""
    e = env.power_task
    robot = env.scene["robot"]
    ids = e._arm_joint_ids
    joint_pos = robot.data.joint_pos[:, ids]
    default_pos = robot.data.default_joint_pos[:, ids]
    joint_vel = robot.data.joint_vel[:, ids]
    return torch.cat((joint_pos - default_pos, joint_vel * 0.1, e.actions_norm), dim=-1)


def ee_and_target(env) -> torch.Tensor:
    """末端位置(3) + 目標-末端誤差(3) + 當前目標點(3) = 9（基座座標系）。"""
    e = env.power_task
    ee_pos_b, _ = e.ee_pose_b()
    target = e.current_target()[:, 0:3]
    return torch.cat((ee_pos_b, target - ee_pos_b, target), dim=-1)


def task_progress(env) -> torch.Tensor:
    """路徑點進度(1) + 末端負載重量(1) = 2。"""
    e = env.power_task
    wp_frac = (e.wp_idx.float() / e.num_wp).unsqueeze(-1)
    payload = torch.full((env.num_envs, 1), e.cfg.payload_mass, device=env.device)
    return torch.cat((wp_frac, payload), dim=-1)
