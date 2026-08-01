# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """計算末端執行器 (EE) 與目標物體之間的相對距離向量 (X, Y, Z)。"""
    """The distance between the end-effector and the object."""
    # 取得末端執行器 (例如夾爪中心) 的座標資料
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    # 取得目標物體 (例如積木) 的物理狀態資料
    object_data: ArticulationData = env.scene["object"].data

    # object_data.root_pos_w 是物體的世界座標
    # ee_tf_data.target_pos_w[..., 0, :] 是末端執行器的世界座標
    # 相減得到相對向量 (Delta X, Delta Y, Delta Z)
    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_ee_flower_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """計算末端執行器與抽屜把手之間的相對距離向量 (X, Y, Z)。"""
    """The distance between the end-effector and the object."""

    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    # 取得抽屜 (或把手) 的座標轉換資料
    flower_tf_data: FrameTransformerData = env.scene["flower_frame"].data

    # [..., 0, :] 的意思是：
    # ... 代表所有的平行環境 (例如 4096 個)
    # 0   代表第一個追蹤點 (通常 0 是夾爪中心，1 和 2 是指尖)
    # :   代表 X, Y, Z 三個座標軸
    return flower_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    """計算指尖相對於該環境原點 (Environment Origins) 的位置。"""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data

    # target_pos_w[..., 1:, :] 的 "1:" 代表取第 1 個之後的所有追蹤點 (通常就是所有的指尖)
    # 減去 env.scene.env_origins 是為了把「世界座標」轉換成「相對於該桌子的局部座標」
    # unsqueeze(1) 是為了讓矩陣維度對齊，才能正確相減
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

    # view(env.num_envs, -1) 會把多個指尖的座標攤平 (Flatten) 成一條長長的一維陣列，方便神經網路讀取
    return fingertips_pos.view(env.num_envs, -1)

def flower_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """花朵相對於該環境原點的位置 (X, Y, Z)。"""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    from isaaclab.assets import RigidObject
    flower: RigidObject = env.scene["flower"]
    # 世界座標 - 環境原點 = 局部座標（每個環境獨立的相對位置）
    return flower.data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    """計算末端執行器相對於該環境原點的位置。"""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    # 直接用 EE 的世界座標 減去 該環境的原點座標
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def rel_flower_bottle_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """花朵中心到瓶口（bottle_top, index 0）的相對向量 (X, Y, Z)。

    第三階段時讓 RL 知道花朵距離瓶口的方向與距離。
    """
    flower_pos  = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    bottle_top  = env.scene["bottle_frame"].data.target_pos_w[..., 0, :]
    return bottle_top - flower_pos


def s1_check_values(env: ManagerBasedRLEnv) -> torch.Tensor:
    """回傳 S1 檢查的數值：[距離, 角度餘弦值]。僅供除錯。"""
    # 距離
    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    bottle_top = env.scene["bottle_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(bottle_top - flower_pos, dim=-1, p=2, keepdim=True)

    # 角度
    flower_quat = env.scene["flower_frame"].data.target_quat_w[..., 0, :]
    flower_rot_mat = math_utils.matrix_from_quat(flower_quat)
    flower_x = flower_rot_mat[..., 0]
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(flower_x)
    align_cosine = (flower_x * world_up).sum(dim=-1, keepdim=True)

    return torch.cat([distance, align_cosine], dim=-1)


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.
    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """

    """取得末端執行器在環境中的旋轉姿態 (以四元數 Quaternion 表示)。
    如果 `make_quat_unique` 為 True，會強制將四元數的實部變為正數，確保數學表達的唯一性。
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data

    # 取得四元數 (w, x, y, z)
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    # 四元數有一個特性：q 和 -q 代表完全一樣的 3D 旋轉。
    # 為了避免神經網路感到困惑 (同一個姿勢卻有兩種不同的輸入數值)，
    # 這裡使用 quat_unique 強制統一格式，這對 AI 學習非常有幫助！
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat
