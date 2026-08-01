# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##==============================================================================================
## 無階段版本（none-stage）：移除事件驅動狀態機，所有獎勵每步一起計算
##==============================================================================================

# 全域步數基準（供 complete_bonus 速度計分用）
_TOTAL_STEPS = 600


def _fire_once(env: ManagerBasedRLEnv, flag_name: str, condition: torch.Tensor) -> torch.Tensor:
    """一次性觸發：條件在該回合首次成立的那一步回傳 True，之後不再觸發。"""
    if not hasattr(env, flag_name):
        setattr(env, flag_name, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    flag = getattr(env, flag_name)

    # 新回合重置（episode_length_buf 在 reward 計算前已 +1，故第一步 = 1）
    flag[env.episode_length_buf == 1] = False

    fire = condition & ~flag
    flag[fire] = True
    return fire


def _remaining_steps(env: ManagerBasedRLEnv) -> torch.Tensor:
    """全域剩餘步數（越早完成剩越多），作為 complete_bonus 的速度計分基準。"""
    return (_TOTAL_STEPS - env.episode_length_buf).clamp(min=0).float()


##==============================================================================================
## first stage
##==============================================================================================

def s0_approach_flower(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    r"""Reward the robot for reaching the flower flower using inverse-square law.

    It uses a piecewise function to reward the robot for reaching the flower.

    .. math::

        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{otherwise}
        \end{cases}

    """
    r"""靠近把手獎勵：使用平方反比定律鼓勵機器人末端執行器（EE）接近抽屜把手。

    使用分段函數來計算獎勵，當距離小於門檻值（threshold）時，獎勵會翻倍。

    數學公式：
    .. math::
        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{如果 距離 } \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{其他}
        \end{cases}
    """


    # 取得 panda_hand 工具中心世界座標（index 0 = ee_tcp）
    tool_center_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]

    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]

    # 計算 tool_center 與花朵的距離
    distance = torch.norm(flower_pos - tool_center_pos, dim=-1, p=2)

    # 計算基礎獎勵值
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 4)

    # 如果距離進入門檻範圍內，給予 2 倍獎勵
    reward = torch.where(distance <= threshold, 2 * reward, reward)

    return reward


def s0_align_flower(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning the end-effector with the flower.

    The reward is based on the alignment of the gripper with the flower. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the flower
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the flower.
    """

    """姿態對齊獎勵：獎勵末端執行器與把手的角度對齊。

    獎勵取決於夾爪與把手的軸向一致性。計算方式如下：
    .. math::
        reward = 0.5 * (align_z^2 + align_x^2)

    其中 align_z 是夾爪 Z 軸與把手 -X 軸的點積，
    align_x 是夾爪 X 軸與把手 -Y 軸的點積。
    """


    # 取得 panda_hand 工具中心方向（index 0 = ee_tcp）
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    flower_quat = env.scene["flower_frame"].data.target_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    flower_mat = matrix_from_quat(flower_quat)

    # tool_center X 軸追蹤 flower X 軸（平行或反平行皆給分）
    ee_tcp_x = ee_tcp_rot_mat[..., 0]
    flower_x = flower_mat[..., 0]

    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), flower_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    reward = align_x ** 4

    return reward


def s0_grasp_flower(
    env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the flower.

    The :attr:`threshold` is the distance from the flower at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    """抓取動作獎勵：當距離把手夠近時，獎勵「閉合手指」的動作。

    :attr:`threshold` 是距離門檻。
    :attr:`open_joint_pos` 是夾爪全開時的關節位置。

    註：這裡假設關節位置為 0 時代表手指完全閉合。
    """


    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    # 取得夾爪目前的關節位置
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    # 取得兩指尖位置，計算中心點
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]
    fingers_mid = (lfinger_pos + rfinger_pos) / 2.0

    # 計算兩指中心點與花朵的距離
    distance = torch.norm(flower_pos - fingers_mid, dim=-1, p=2)
    is_close = distance <= threshold

    # ── 角度條件：兩指連線與花朵 X 軸夾角 ≥ 45° 才允許給分 ──
    # 取得花朵 X 軸
    flower_quat = env.scene["flower_frame"].data.target_quat_w[..., 0, :]
    flower_rot_mat = matrix_from_quat(flower_quat)
    flower_x = flower_rot_mat[..., 0]  # (num_envs, 3)
    finger_vec = rfinger_pos - lfinger_pos
    finger_vec_normalized = finger_vec / (torch.norm(finger_vec, dim=-1, keepdim=True) + 1e-6)

    # |cos θ| ≤ cos(45°) ≈ 0.707 → 夾角 ≥ 45°
    cos_angle = torch.sum(finger_vec_normalized * flower_x, dim=-1)
    is_aligned = torch.abs(cos_angle) <= 0.707  # 夾角在 45°~135° 之間

    reward = torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)

    # 同時滿足：距離夠近 AND 方向正確，才獎勵閉合
    reward = is_close * is_aligned * reward

    return reward


def s0_lift_when_grasped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """夾緊後提升獎勵：當兩支夾爪間距小於 1.5cm 時，獎勵 EE 的絕對高度。

    每提高 1mm 給 0.01 分，上限 0.2 分。夾爪間距大於 1.5cm 時歸零。
    """


    # 取得夾爪目前的關節位置
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    # 兩根手指各自代表一半的開合量，相加即為夾爪總間距
    gripper_gap = torch.sum(gripper_joint_pos, dim=-1)  # 單位：公尺

    # 夾爪間距 < 1.5cm 才觸發
    is_closed = (gripper_gap < 0.03).float()

    # 取得兩指尖中心點與花朵的距離，< 2cm 才觸發
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]
    fingers_mid = (lfinger_pos + rfinger_pos) / 2.0
    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    flower_dist = torch.norm(flower_pos - fingers_mid, dim=-1, p=2)
    is_flower_grasped = (flower_dist < 0.03).float()

    # 取得 EE 世界座標 Z 高度（距離地面高度）
    ee_pos_w = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    ee_height = ee_pos_w[:, 2]  # 單位：公尺

    # 每 1mm (0.001m) = 0.01 分，換算係數 = 10
    # 上限 0.2 分（即 EE 高度 2cm 以上即封頂）
    reward = torch.clamp(ee_height * 10.0, min=0.0, max=0.2)
    apply_reward = is_closed * is_flower_grasped * reward

    return apply_reward





def s0_multi_lift(env: ManagerBasedRLEnv) -> torch.Tensor:
    """多階段拾起獎勵：將拾起花朵任務分為「簡單、中等、困難」三個層次。

    這能幫助 Agent（AI）循序漸進地學會抬高花朵。
    """
    """Multi-stage bonus for lifting the flower.

    Depending on the flower's lift height, the reward is given in three stages: easy, medium, and hard.
    This helps the agent to learn to pick up the flower in a controlled manner.
    """

    flower_pos_w = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    caught       = is_catch(env)
    is_graspable = align_grasp_around_flower(env).float()
    lift_height  = torch.clamp(flower_pos_w[:, 2] - 0.03, min=0.0)


    # 第一階段：只要花朵有離地超過 0.04m 就給 30 分
    open_easy = (lift_height > 0.04)
    open_easy_catch = (lift_height > 0.04)  * is_graspable * caught
    # 第二階段：抬高超過 0.1m 且手姿勢正確，給 50 分
    open_medium = (lift_height > 0.1)
    open_medium_catch = (lift_height > 0.1) * is_graspable * caught

    # 第二階段：抬高超過 0.1m 且手姿勢正確，給 50 分
    open_hard = (lift_height > 0.3)
    open_hard_catch = (lift_height > 0.3) * is_graspable * caught


    reward = open_easy + open_easy_catch + open_medium_catch + open_hard_catch

    return reward


def s0_complete_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """S0 完成條件首次成立時給予速度獎勵（全域剩餘步數），並觸發瓶子瞬移。"""
    fire = _fire_once(env, "_s0_bonus_fired", _first_stage_complete(env).bool())

    # 瓶子瞬移：S0 完成當步將瓶子移到花朵的 x, y（z = 0）
    spawn_ids = torch.where(fire)[0]
    if len(spawn_ids) > 0:
        bottle = env.scene["bottle"]
        flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
        root_state = bottle.data.root_state_w[spawn_ids].clone()
        root_state[:, 0] = flower_pos[spawn_ids, 0]
        root_state[:, 1] = flower_pos[spawn_ids, 1]
        root_state[:, 2] = 0.0
        root_state[:, 7:] = 0.0
        bottle.write_root_state_to_sim(root_state, env_ids=spawn_ids)

    return fire.float() * _remaining_steps(env)


def s0_catch(env: ManagerBasedRLEnv) -> torch.Tensor:
    """tool_center 與花朵直線距離 < 0.003m 時，每 tick 給 3 分。"""
    tool_center_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(flower_pos - tool_center_pos, dim=-1, p=2)
    return (distance < 0.003).float() * 3.0


def s0_touch_flower(env: ManagerBasedRLEnv) -> torch.Tensor:
    """夾住花朵時每 tick 給分。"""
    return is_catch(env)


def s1_arm_hold(env: ManagerBasedRLEnv) -> torch.Tensor:
    """手臂關節速度接近零（維持不動）→ 獎勵。"""
    robot = env.scene["robot"]
    arm_ids = robot.find_joints("panda_joint.*")[0]
    vel = robot.data.joint_vel[:, arm_ids]
    return (torch.norm(vel, dim=-1) < 0.1).float()


def s1_gripper_hold(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """夾爪關節速度接近零（維持不動）→ 獎勵。"""
    gripper_vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return (torch.norm(gripper_vel, dim=-1) < 0.01).float()


def s1_approach_bottle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """引導 tool_center 移動到瓶口（bottle_top）位置，距離平方反比 n=4。"""
    tool_center_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    bottle_top      = env.scene["bottle_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(bottle_top - tool_center_pos, dim=-1, p=2)
    return torch.pow(1.0 / (1.0 + distance**2), 4)


def s1_align_flower_up(env: ManagerBasedRLEnv) -> torch.Tensor:
    """花朵 +X 軸對齊世界 +Z（朝上），n=4 曲率。"""
    flower_quat    = env.scene["flower_frame"].data.target_quat_w[..., 0, :]
    flower_rot_mat = matrix_from_quat(flower_quat)
    flower_x = flower_rot_mat[..., 0]          # (num_envs, 3)
    world_up = torch.zeros_like(flower_x)
    world_up[:, 2] = 1.0                       # (0, 0, 1)
    align  = (flower_x * world_up).sum(dim=-1).clamp(min=0.0)
    return torch.pow(align, 4)


def _is_s1_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """輔助函數：檢查 S1 任務是否完成。

    完成條件 (兩項皆成立)：
    1. 花朵與瓶口距離 <= 5cm
    2. 花朵 X 軸與世界 Z 軸夾角 <= 10 度
    """
    # 距離檢查
    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    bottle_top = env.scene["bottle_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(bottle_top - flower_pos, dim=-1, p=2)
    dist_ok = distance <= 0.05

    # 角度檢查 (cos(10 deg) approx 0.985)
    flower_quat = env.scene["flower_frame"].data.target_quat_w[..., 0, :]
    flower_rot_mat = matrix_from_quat(flower_quat)
    flower_x = flower_rot_mat[..., 0]
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(flower_x)
    align_cosine = (flower_x * world_up).sum(dim=-1)
    angle_ok = align_cosine >= 0.985

    return dist_ok & angle_ok

def s1_complete_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """S1 完成條件首次成立時給予速度獎勵（全域剩餘步數）。"""
    fire = _fire_once(env, "_s1_bonus_fired", _is_s1_complete(env))
    return fire.float() * _remaining_steps(env)


def s1_dead_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """無階段版本：不再有階段逾時死亡，恆為 False。"""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def s1_dead_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """無階段版本：不再有階段逾時懲罰，恆為 0。"""
    return torch.zeros(env.num_envs, device=env.device)


def phase0_dead_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """無階段版本：不再有階段逾時死亡，恆為 False。"""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def dead_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """無階段版本：不再有階段逾時懲罰，恆為 0。"""
    return torch.zeros(env.num_envs, device=env.device)


##==============================================================================================
## third stage
##==============================================================================================

def s2_approach_inside(env: ManagerBasedRLEnv) -> torch.Tensor:
    """花朵靠近瓶子原點，距離越近分數越高。"""
    bottle_origin = env.scene["bottle"].data.root_pos_w
    flower_pos    = env.scene["flower_frame"].data.target_pos_w[..., 0, :]

    distance = torch.norm(bottle_origin - flower_pos, dim=-1, p=2)
    reward   = 1.0 / (1.0 + (distance * 5.0) ** 2)
    return torch.pow(reward, 3)


def s2_release(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """放手獎勵：獎勵張開夾爪，夾爪越開分數越高。"""
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(gripper_joint_pos, dim=-1)


def _is_s2_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 2 完成條件：花朵位於 bottle_inside 下方且 x/y 對齊（< 2cm），
    且花朵 X 軸與世界 Z 軸夾角 <= 35 度。
    """
    bottle_inside = env.scene["bottle_frame"].data.target_pos_w[..., 1, :]
    flower_pos    = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    below  = flower_pos[:, 2] < bottle_inside[:, 2]
    near_x = torch.abs(flower_pos[:, 0] - bottle_inside[:, 0]) < 0.02
    near_y = torch.abs(flower_pos[:, 1] - bottle_inside[:, 1]) < 0.02

    # 花朵 +X 朝上檢查（cos(35 deg) approx 0.819）
    flower_quat = env.scene["flower_frame"].data.target_quat_w[..., 0, :]
    flower_rot_mat = matrix_from_quat(flower_quat)
    flower_x = flower_rot_mat[..., 0]
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(flower_x)
    align_cosine = (flower_x * world_up).sum(dim=-1)
    angle_ok = align_cosine >= 0.819

    return below & near_x & near_y & angle_ok


def s2_complete_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """S2 完成條件首次成立時給予速度獎勵（全域剩餘步數）。"""
    fire = _fire_once(env, "_s2_bonus_fired", _is_s2_complete(env))
    return fire.float() * _remaining_steps(env)


def all_complete_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """全任務完成（S2 完成當步）之獨立速度獎勵，與 s2_complete_bonus 各自加總。"""
    fire = _fire_once(env, "_all_bonus_fired", _is_s2_complete(env))
    return fire.float() * _remaining_steps(env)


def s2_dead_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """無階段版本：不再有階段逾時死亡，恆為 False。"""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def s2_dead_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """無階段版本：不再有階段逾時懲罰，恆為 0。"""
    return torch.zeros(env.num_envs, device=env.device)


def task_success_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """S2 完成條件成立 → 任務成功，結束回合。"""
    return _is_s2_complete(env)


##==============================================================================================
## 第一階段完成條件（內部輔助）
##==============================================================================================

def _first_stage_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """第一階段完成：與 multi_stage_pick_up 的 open_hard_catch 相同條件。

    同時滿足：
    - 花朵高度 > 0.3m（lift_height > 0.3）
    - 兩指在花朵 5cm 以內（is_graspable）
    - 任一手指施力 >= 1N 且靠近花朵（caught）
    """
    flower_pos_w = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    lift_height  = torch.clamp(flower_pos_w[:, 2] - 0.03, min=0.0)

    caught       = is_catch(env)
    is_graspable = align_grasp_around_flower(env)

    open_hard_catch = (lift_height > 0.3) & is_graspable & caught.bool()

    return open_hard_catch.float()


##==============================================================================================
## 內部輔助函式
##==============================================================================================


def align_grasp_around_flower(env: ManagerBasedRLEnv) -> torch.Tensor:
    """兩指是否同時在花朵 5cm 以內（bool tensor）。內部輔助用。"""
    flower_pos      = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos     = ee_fingertips_w[..., 0, :]
    rfinger_pos     = ee_fingertips_w[..., 1, :]

    lfinger_dist = torch.norm(lfinger_pos - flower_pos, dim=-1, p=2)
    rfinger_dist = torch.norm(rfinger_pos - flower_pos, dim=-1, p=2)

    return (lfinger_dist < 0.05) & (rfinger_dist < 0.05)


def is_catch(env: ManagerBasedRLEnv) -> torch.Tensor:
    """偵測任一手指在夾爪局部 X 軸方向施加 1N 以上的力，且指尖距花朵 < 5cm。

    Returns:
        shape (num_envs,) 的 float tensor：1.0 = 抓住，0.0 = 未抓住
    """
    contact_sensor = env.scene["contact_forces"]
    net_forces     = contact_sensor.data.net_forces_w.clamp(-100.0, 100.0)  # (num_envs, 2, 3)

    finger_quats   = env.scene["ee_frame"].data.target_quat_w[..., 1:, :]   # (num_envs, 2, 4)
    finger_rot_mat = matrix_from_quat(finger_quats.reshape(-1, 4)).reshape(-1, 2, 3, 3)
    local_x        = finger_rot_mat[..., 0]                                  # (num_envs, 2, 3)

    force_along_x  = torch.abs((net_forces * local_x).sum(dim=-1))           # (num_envs, 2)

    flower_pos      = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos     = ee_fingertips_w[..., 0, :]
    rfinger_pos     = ee_fingertips_w[..., 1, :]
    ldist           = torch.norm(lfinger_pos - flower_pos, dim=-1)
    rdist           = torch.norm(rfinger_pos - flower_pos, dim=-1)

    finger_near_flower = (ldist < 0.05) | (rdist < 0.05)
    any_finger_force   = force_along_x.max(dim=-1).values >= 1.0

    return (any_finger_force & finger_near_flower).float()


# ──────────────────────────────────────────────────────────────────────────────
# 動作平滑懲罰（無階段版本：全程一致，不再依階段加權）
# ──────────────────────────────────────────────────────────────────────────────

def all_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """action_rate_l2，全程一致。"""
    from isaaclab.envs.mdp.rewards import action_rate_l2
    return action_rate_l2(env)


def all_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """joint_vel_l2，全程一致。"""
    from isaaclab.envs.mdp.rewards import joint_vel_l2
    return joint_vel_l2(env, asset_cfg)
