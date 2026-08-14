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

def _compute_stage_tag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """計算當前任務階段標籤（純時間驅動），shape (num_envs, 4)，One-Hot 編碼。

    0–5s   (step   0–299): Stage 0 — 接近、抓取、舉起花朵
    5–7s   (step 300–419): Stage 1 — 移動到瓶口
    7–10s  (step 420–599): Stage 2 — 插入瓶子
    """
    step    = env.episode_length_buf
    phase1 = int(5.0 / env.step_dt)   # 300
    phase2 = int(7.0 / env.step_dt)   # 420

    # 用 > 避免邊界步同時觸發當前階段獎勵與死亡判斷
    stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    stage[step > phase1] = 1
    stage[step > phase2] = 2

    tag = torch.zeros(env.num_envs, 4, device=env.device)
    tag.scatter_(1, stage.unsqueeze(1), 1.0)
    return tag


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

    tag = _compute_stage_tag(env)
    return tag[:, 0] * reward


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

    # 兩個條件各佔 0.5，合計最高 1 分
    tag = _compute_stage_tag(env)
    return tag[:, 0] * reward


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

    tag = _compute_stage_tag(env)
    return tag[:, 0] * reward


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

    tag = _compute_stage_tag(env)
    return tag[:, 0] * apply_reward





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

    tag = _compute_stage_tag(env)
    return tag[:, 0] * reward


def s0_complete_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """第一階段首次完成：速度獎勵（越早完成分越高）＋瓶子瞬移。

    bonus = weight × (1 - step / max_episode_length)
    """
    if not hasattr(env, "_s0_bonus_given_flag"):
        env._s0_bonus_given_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # episode_length_buf 在 reward 計算前已 +1，所以新回合第一步 = 1
    reset_mask = env.episode_length_buf == 1
    env._s0_bonus_given_flag[reset_mask] = False

    tag = _compute_stage_tag(env)
    complete = _first_stage_complete(env).bool() & tag[:, 0].bool()
    fire = complete & ~env._s0_bonus_given_flag
    env._s0_bonus_given_flag[fire] = True

    # 瓶子瞬移（同一步觸發，無延遲）
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

    # 速度獎勵
    time_left_frac = (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(min=0.0)
    return fire.float() * time_left_frac


def s0_catch(env: ManagerBasedRLEnv) -> torch.Tensor:
    """夾住花朵時每 tick 給分（第一階段限定）。"""
    tag = _compute_stage_tag(env)
    return tag[:, 0] * is_catch(env)


def s0_touch_flower(env: ManagerBasedRLEnv) -> torch.Tensor:
    """tool_center 與花朵直線距離 < 0.003m 時，每 tick 給 3 分（第一階段限定）。"""
    tag = _compute_stage_tag(env)
    tool_center_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(flower_pos - tool_center_pos, dim=-1, p=2)
    return tag[:, 0] * (distance < 0.003).float() * 3.0


def s1_arm_hold(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 2：手臂關節速度接近零（維持不動）→ 獎勵。"""
    tag = _compute_stage_tag(env)
    robot = env.scene["robot"]
    arm_ids = robot.find_joints("panda_joint.*")[0]
    vel = robot.data.joint_vel[:, arm_ids]
    held_still = (torch.norm(vel, dim=-1) < 0.1).float()
    return tag[:, 1] * held_still


def s1_gripper_hold(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Stage 2：夾爪關節速度接近零（維持不動）→ 獎勵。"""
    tag = _compute_stage_tag(env)
    gripper_vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    held_still = (torch.norm(gripper_vel, dim=-1) < 0.01).float()
    return tag[:, 1] * held_still


def s1_approach_bottle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 1：引導 tool_center 移動到瓶口（bottle_top）位置，距離平方反比 n=4。"""
    tag = _compute_stage_tag(env)
    tool_center_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    bottle_top      = env.scene["bottle_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(bottle_top - tool_center_pos, dim=-1, p=2)
    reward = torch.pow(1.0 / (1.0 + distance**2), 4)
    return tag[:, 1] * reward


def s1_align_flower_up(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 1：花朵 +X 軸對齊世界 +Z（朝上），n=4 曲率。"""
    tag = _compute_stage_tag(env)
    flower_quat    = env.scene["flower_frame"].data.target_quat_w[..., 0, :]
    flower_rot_mat = matrix_from_quat(flower_quat)
    flower_x = flower_rot_mat[..., 0]          # (num_envs, 3)
    world_up = torch.zeros_like(flower_x)
    world_up[:, 2] = 1.0                       # (0, 0, 1)
    align  = (flower_x * world_up).sum(dim=-1).clamp(min=0.0)
    reward = torch.pow(align, 4)
    return tag[:, 1] * reward


def _is_s1_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """輔助函數：檢查 S1 任務是否完成。

    完成條件 (兩項皆成立)：
    1. 花朵與瓶口距離 <= 5cm
    2. 花朵 X 軸與世界 Z 軸夾角 <= 35 度
    """
    # 距離檢查
    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    bottle_top = env.scene["bottle_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(bottle_top - flower_pos, dim=-1, p=2)
    dist_ok = distance <= 0.05

    # 角度檢查 (cos(35 deg) approx 0.819)
    flower_quat = env.scene["flower_frame"].data.target_quat_w[..., 0, :]
    flower_rot_mat = matrix_from_quat(flower_quat)
    flower_x = flower_rot_mat[..., 0]
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(flower_x)
    align_cosine = (flower_x * world_up).sum(dim=-1)
    angle_ok = align_cosine >= 0.819

    return dist_ok & angle_ok

def _is_s1_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """輔助函數：檢查是否「時間已到 7 秒」且「S1 任務失敗」。

    失敗條件 (任一項成立)：
    1. 花朵與瓶口距離 > 5cm
    2. 花朵 X 軸與世界 Z 軸夾角 > 35 度
    """
    phase2_steps = int(7.0 / env.step_dt)
    at_boundary = env.episode_length_buf >= phase2_steps

    # 如果 S1 已經成功，就不算失敗
    # 防禦性檢查：如果終止條件在獎勵函數前執行，_s2_bonus_given 可能不存在
    if not hasattr(env, "_s1_bonus_given_flag"):
        return at_boundary  # 假設 S1 尚未成功，如果時間到就判定失敗
    s1_was_completed = env._s1_bonus_given_flag

    failed = at_boundary & ~s1_was_completed
    return failed


def s1_dead_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """S1 結束時，若未達成條件則死亡。"""
    return _is_s1_failed(env)


def s1_dead_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """當 s1_dead_termination 觸發時，給予一次性懲罰。"""
    return _is_s1_failed(env).float()


def s1_complete_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """S1 階段首次完成：給予速度獎勵（越早完成分數越高）。"""
    # 初始化旗標，用於記錄獎勵是否已發放
    if not hasattr(env, "_s1_bonus_given_flag"):
        env._s1_bonus_given_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 回合重置時，清除旗標
    reset_mask = env.episode_length_buf == 1
    env._s1_bonus_given_flag[reset_mask] = False

    # 檢查是否在 S1 階段且任務完成
    tag = _compute_stage_tag(env)
    in_stage1 = tag[:, 1].bool()
    is_complete = _is_s1_complete(env)

    # 找出首次完成的環境 (in_stage1 AND is_complete AND not bonus_given)
    fire = in_stage1 & is_complete & ~env._s1_bonus_given_flag
    env._s1_bonus_given_flag[fire] = True

    # 計算速度獎勵
    time_left_frac = (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(min=0.0)
    return fire.float() * time_left_frac

def _is_phase0_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """輔助函數：檢查是否「時間已到 5 秒」且「第一階段未完成」"""
    phase1_steps = int(5.0 / env.step_dt)   # 300
    at_boundary  = env.episode_length_buf >= phase1_steps

    if not hasattr(env, "_s0_bonus_given_flag"):
        env._s0_bonus_given_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return at_boundary & ~env._s0_bonus_given_flag


def phase0_dead_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phase 0（0–5s）時限屆滿且未完成第一階段 → 死亡。"""
    return _is_phase0_failed(env)


def dead_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """當 phase0_dead_termination 觸發時，給予一次性懲罰。"""
    return _is_phase0_failed(env).float()


def stage2_low_flower_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 1（等待期）期間，花朵高度 < 0.3m 時每 tick 給予懲罰。"""
    tag = _compute_stage_tag(env)
    flower_pos_w = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    lift_height = flower_pos_w[:, 2] - 0.03
    return (tag[:, 1].bool() & (lift_height < 0.3)).float()


def stage2_low_flower_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 1（等待期）期間，只要花朵高度 < 0.3m 立即終止回合。"""
    tag = _compute_stage_tag(env)
    flower_pos_w = env.scene["flower_frame"].data.target_pos_w[..., 0, :]
    lift_height = flower_pos_w[:, 2] - 0.03
    return tag[:, 1].bool() & (lift_height < 0.3)


def teleport_bottle_on_stage1_complete(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
) -> None:
    """進入 Stage 2（等待期）時，將瓶子從遠處召喚到花朵的 x, y, z=0（僅一次）。"""
    # 延遲初始化旗標，用於記錄瓶子是否已被召喚
    if not hasattr(env, "_bottle_is_spawned"):
        env._bottle_is_spawned = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 環境重置時，清除召喚旗標
    reset_mask = env.episode_length_buf == 0
    env._bottle_is_spawned[reset_mask] = False

    tag      = _compute_stage_tag(env)
    in_stage2 = tag[:, 1].bool()   # 已進入第二階段（等待期）
    bottle   = env.scene["bottle"]

    # 只對「已進入 Stage 2 且 瓶子尚未召喚」的環境執行
    needs_spawn  = in_stage2 & ~env._bottle_is_spawned
    spawn_ids    = torch.where(needs_spawn)[0]

    if len(spawn_ids) == 0:
        return

    flower_pos = env.scene["flower_frame"].data.target_pos_w[..., 0, :]

    root_state = bottle.data.root_state_w[spawn_ids].clone()
    root_state[:, 0] = flower_pos[spawn_ids, 0]   # x from flower
    root_state[:, 1] = flower_pos[spawn_ids, 1]   # y from flower
    root_state[:, 2] = 0.0                         # z = 0
    root_state[:, 7:] = 0.0                        # zero velocity

    bottle.write_root_state_to_sim(root_state, env_ids=spawn_ids)
    env._bottle_is_spawned[spawn_ids] = True       # 設定旗標，防止重複召喚




##==============================================================================================
## third stage  （僅狀態 [0, 0, 1, 0] 有效，tag[:, 2] == 1）
##==============================================================================================

def s2_approach_inside(env: ManagerBasedRLEnv) -> torch.Tensor:
    """第三階段獎勵：tool_center 靠近瓶子原點，距離越近分數越高。"""
    bottle_origin    = env.scene["bottle"].data.root_pos_w
    tool_center_pos  = env.scene["ee_frame"].data.target_pos_w[..., 0, :]

    distance = torch.norm(bottle_origin - tool_center_pos, dim=-1, p=2)
    reward   = 1.0 / (1.0 + (distance * 5.0) ** 2)
    reward   = torch.pow(reward, 3)

    tag = _compute_stage_tag(env)
    return tag[:, 2] * reward


def s2_release(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """第三階段放手獎勵：獎勵張開夾爪，夾爪越開分數越高。"""
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(gripper_joint_pos, dim=-1)

    tag = _compute_stage_tag(env)
    return tag[:, 2] * reward


def s2_complete_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """第三階段首次完成：速度獎勵（越早完成分越高）。

    bonus = weight × (1 - step / max_episode_length)
    """
    if not hasattr(env, "_s3_bonus_given"):
        env._s3_bonus_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # episode_length_buf 在 reward 計算前已 +1，所以新回合第一步 = 1
    reset_mask = env.episode_length_buf == 1
    env._s3_bonus_given[reset_mask] = False

    bottle_inside   = env.scene["bottle_frame"].data.target_pos_w[..., 1, :]
    tool_center_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    below  = (tool_center_pos[:, 2] < bottle_inside[:, 2])
    near_x = (torch.abs(tool_center_pos[:, 0] - bottle_inside[:, 0]) < 0.02)
    near_y = (torch.abs(tool_center_pos[:, 1] - bottle_inside[:, 1]) < 0.02)
    complete = below & near_x & near_y

    tag = _compute_stage_tag(env)
    fire = complete & tag[:, 2].bool() & ~env._s3_bonus_given
    env._s3_bonus_given[fire] = True

    time_left_frac = (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(min=0.0)
    return fire.float() * time_left_frac


def all_complete_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """全任務完成（Stage 2 完成當步）獨立速度獎勵，觸發條件與 s2_complete_bonus 相同。

    bonus = weight × (1 - step / max_episode_length)，與 s2_complete_bonus 同時發放、互不取代。
    """
    if not hasattr(env, "_all_bonus_given"):
        env._all_bonus_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    reset_mask = env.episode_length_buf == 1
    env._all_bonus_given[reset_mask] = False

    bottle_inside   = env.scene["bottle_frame"].data.target_pos_w[..., 1, :]
    tool_center_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    below  = (tool_center_pos[:, 2] < bottle_inside[:, 2])
    near_x = (torch.abs(tool_center_pos[:, 0] - bottle_inside[:, 0]) < 0.02)
    near_y = (torch.abs(tool_center_pos[:, 1] - bottle_inside[:, 1]) < 0.02)
    complete = below & near_x & near_y

    tag = _compute_stage_tag(env)
    fire = complete & tag[:, 2].bool() & ~env._all_bonus_given
    env._all_bonus_given[fire] = True

    time_left_frac = (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(min=0.0)
    return fire.float() * time_left_frac


##==============================================================================================
## 第一階段完成條件（內部，供 _compute_stage_tag 呼叫）
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
# Stage 2 加重版懲罰（第二階段額外乘 10 倍，不限 epoch）
# ──────────────────────────────────────────────────────────────────────────────

def all_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """action_rate_l2，第二階段額外乘 100 倍。"""
    from isaaclab.envs.mdp.rewards import action_rate_l2
    base      = action_rate_l2(env)
    in_stage2 = _compute_stage_tag(env)[:, 1]
    return base * (1.0 + 99.0 * in_stage2)   # stage2 → ×100，其餘 → ×1


def all_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """joint_vel_l2，第二階段額外乘 100 倍。"""
    from isaaclab.envs.mdp.rewards import joint_vel_l2
    base      = joint_vel_l2(env, asset_cfg)
    in_stage2 = _compute_stage_tag(env)[:, 1]
    return base * (1.0 + 99.0 * in_stage2)   # stage2 → ×100，其餘 → ×1
