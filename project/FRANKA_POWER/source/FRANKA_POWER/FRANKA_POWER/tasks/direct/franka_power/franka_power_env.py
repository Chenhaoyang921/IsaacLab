# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka 最小功率走點任務（Direct workflow）。

流程：
  1. 差分 IK 依序追蹤使用者在 cfg 中填寫的路徑點，產生關節位置目標。
  2. Agent 每一步輸出 7 個關節的力矩上限（= 各關節的輸出功率額定值）。
  3. IdealPDActuator 依 PD 律計算力矩後，被 Agent 給的上限截斷。
  4. 獎勵 = 任務進度 - 消耗的電能（機械功率 + 銅損發熱）。

Agent 的目標：找出能扛著末端負載走完全部路徑點的「最小」關節輸出功率。
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, sample_uniform, subtract_frame_transforms

from .franka_power_env_cfg import FrankaPowerEnvCfg


class FrankaPowerEnv(DirectRLEnv):
    cfg: FrankaPowerEnvCfg

    def __init__(self, cfg: FrankaPowerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- 關節 / body 索引 ----
        self._arm_joint_ids, self._arm_joint_names = self._robot.find_joints("panda_joint[1-7]")
        self._finger_joint_ids, _ = self._robot.find_joints("panda_finger_joint.*")
        self._hand_body_idx = self._robot.find_bodies("panda_hand")[0][0]
        # 固定基座機器人的 Jacobian 不含 root body，索引要 -1
        self._ee_jacobi_idx = self._hand_body_idx - 1

        # ---- 手臂顯式致動器（Agent 每步改寫其 effort_limit）----
        self._arm_actuator = self._robot.actuators["panda_arm"]

        # ---- 路徑點：解析成 (num_wp, 7) 張量（基座座標系）----
        wp_list = []
        for wp in self.cfg.waypoints:
            if len(wp) == 3:
                wp_list.append(list(wp) + list(self.cfg.default_ee_quat))
            elif len(wp) == 7:
                wp_list.append(list(wp))
            else:
                raise ValueError(f"路徑點必須是 [x,y,z] 或 [x,y,z,qw,qx,qy,qz]，收到: {wp}")
        self._waypoints = torch.tensor(wp_list, dtype=torch.float, device=self.device)
        self._num_wp = self._waypoints.shape[0]

        # ---- 每個路徑點各自的時間預算（超時判定用）----
        if len(self.cfg.waypoint_timeout_s) != self._num_wp:
            raise ValueError(
                f"waypoint_timeout_s 長度 ({len(self.cfg.waypoint_timeout_s)}) 必須跟 "
                f"waypoints 數量 ({self._num_wp}) 一致，兩者要一一對應。"
            )
        self._wp_timeout_steps = torch.tensor(
            [max(1, int(round(t / self.step_dt))) for t in self.cfg.waypoint_timeout_s],
            dtype=torch.long, device=self.device,
        )
        # 抵達時間窗下界（每個路徑點；0 表示沒有下界，不罰早到）
        if len(self.cfg.waypoint_time_min_s) != self._num_wp:
            raise ValueError(
                f"waypoint_time_min_s 長度 ({len(self.cfg.waypoint_time_min_s)}) 必須跟 "
                f"waypoints 數量 ({self._num_wp}) 一致，兩者要一一對應。"
            )
        self._wp_min_steps = torch.tensor(
            [max(0, int(round(t / self.step_dt))) for t in self.cfg.waypoint_time_min_s],
            dtype=torch.long, device=self.device,
        )

        # ---- 差分 IK 控制器 ----
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method=self.cfg.ik_method
        )
        self._diff_ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # ---- 力矩上限搜尋範圍 ----
        self._effort_min = torch.tensor(self.cfg.effort_limit_min, device=self.device)
        self._effort_max = torch.tensor(self.cfg.effort_limit_max, device=self.device)

        # ---- 末端負載：把模擬物品的質量（與等比例慣量）加到 panda_hand ----
        if self.cfg.payload_mass > 0.0:
            self._apply_payload(self.cfg.payload_mass)

        # ---- 狀態緩衝 ----
        self.actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._prev_actions = torch.zeros_like(self.actions)
        self._effort_limits = self._effort_max.unsqueeze(0).repeat(self.num_envs, 1)
        self._joint_pos_des = self._robot.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_dist = torch.full((self.num_envs,), -1.0, device=self.device)  # <0 表示剛重置
        self._dist = torch.zeros(self.num_envs, device=self.device)
        self._progress = torch.zeros(self.num_envs, device=self.device)
        self._reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._task_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._task_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._power_W = torch.zeros(self.num_envs, device=self.device)

        # ---- 每個路徑點的時間預算（超時判定，計數器）----
        self._wp_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._wp_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._window_pen = torch.zeros(self.num_envs, device=self.device)  # 抵達時間偏離窗口的秒數

        # ---- episode 統計（訓練 log 用）----
        self._ep_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in [
                "progress", "waypoint_bonus", "success_bonus", "window_pen", "dist_pen",
                "energy_pen", "limit_pen", "action_rate_pen", "timeout_pen", "energy_J",
            ]
        }
        self._ep_limit_sums = torch.zeros(self.num_envs, 7, device=self.device)

        # ---- 視覺化（僅在有 GUI 時）----
        self._markers_enabled = self.sim.has_gui()
        if self._markers_enabled:
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
            self._goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/goal"))
            self._ee_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/ee"))

        print(
            f"[FrankaPowerEnv] 路徑點數量: {self._num_wp}, 末端負載: {self.cfg.payload_mass} kg, "
            f"各點時間預算(s): {self.cfg.waypoint_timeout_s} -> steps: {self._wp_timeout_steps.tolist()}"
        )

    # ------------------------------------------------------------------
    # 場景建立
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # 地板
        spawn_ground = sim_utils.GroundPlaneCfg()
        spawn_ground.func("/World/ground", spawn_ground, translation=(0.0, 0.0, 0.0))
        # 複製環境
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # 燈光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_payload(self, payload_mass: float):
        """把負載質量加到 panda_hand，慣量依質量比例放大（點質量近似）。"""
        masses = self._robot.root_physx_view.get_masses().clone()
        inertias = self._robot.root_physx_view.get_inertias().clone()
        hand_mass = masses[:, self._hand_body_idx].clone()
        ratio = (hand_mass + payload_mass) / hand_mass
        masses[:, self._hand_body_idx] += payload_mass
        inertias[:, self._hand_body_idx, :] *= ratio.unsqueeze(-1)
        all_ids = torch.arange(self.num_envs)
        self._robot.root_physx_view.set_masses(masses, all_ids)
        self._robot.root_physx_view.set_inertias(inertias, all_ids)

    # ------------------------------------------------------------------
    # 每個 policy step
    # ------------------------------------------------------------------

    def _ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """末端（panda_hand）在基座座標系的位置與姿態。"""
        ee_pose_w = self._robot.data.body_pose_w[:, self._hand_body_idx]
        root_pose_w = self._robot.data.root_pose_w
        return subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        # ---- 1. Agent 動作 -> 各關節力矩上限（輸出功率）----
        self._prev_actions = self.actions.clone()
        self.actions = actions.clamp(-1.0, 1.0).clone()
        self._effort_limits = self._effort_min + 0.5 * (self.actions + 1.0) * (
            self._effort_max - self._effort_min
        )
        # 直接改寫顯式致動器的力矩上限（(num_envs, 7) 張量）
        self._arm_actuator.effort_limit[:] = self._effort_limits

        # ---- 2. 差分 IK：朝當前路徑點解出關節位置目標 ----
        target_pose = self._waypoints[self._wp_idx]  # (N, 7) 基座座標系
        self._diff_ik.set_command(target_pose)
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, :
        ][:, :, self._arm_joint_ids]
        ee_pos_b, ee_quat_b = self._ee_pose_b()
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        self._joint_pos_des = self._diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # ---- 3. 視覺化 ----
        if self._markers_enabled:
            root_pose_w = self._robot.data.root_pose_w
            goal_pos_w, goal_quat_w = combine_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pose[:, 0:3], target_pose[:, 3:7]
            )
            self._goal_marker.visualize(goal_pos_w, goal_quat_w)
            ee_pose_w = self._robot.data.body_pose_w[:, self._hand_body_idx]
            self._ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

    def _apply_action(self):
        # 手臂：IK 解出的關節位置目標（顯式 PD 會轉成力矩並被上限截斷）
        self._robot.set_joint_position_target(self._joint_pos_des, joint_ids=self._arm_joint_ids)
        # 夾爪：維持預設開度
        self._robot.set_joint_position_target(
            self._robot.data.default_joint_pos[:, self._finger_joint_ids],
            joint_ids=self._finger_joint_ids,
        )

    # ------------------------------------------------------------------
    # 任務狀態（在 _get_dones 中更新，_get_rewards 直接取用）
    # ------------------------------------------------------------------

    def _update_task_state(self):
        self._wp_step_counter += 1

        ee_pos_b, _ = self._ee_pose_b()
        target = self._waypoints[self._wp_idx]
        self._dist = torch.norm(ee_pos_b - target[:, 0:3], dim=-1)

        # 進度 = 上一步距離 - 這一步距離（剛重置的環境進度為 0）
        fresh = self._prev_dist < 0.0
        self._progress = torch.where(
            fresh, torch.zeros_like(self._dist), self._prev_dist - self._dist
        )

        # 到達判定與路徑點推進
        self._reached = self._dist < self.cfg.waypoint_tolerance

        # 抵達時間窗懲罰：用「推進前」的 wp_idx 與計數器，算這次抵達偏離 [MIN, MAX] 幾秒
        min_steps = self._wp_min_steps[self._wp_idx]
        max_steps = self._wp_timeout_steps[self._wp_idx]
        early = (min_steps - self._wp_step_counter).clamp(min=0)
        late = (self._wp_step_counter - max_steps).clamp(min=0)
        self._window_pen = self._reached.float() * (early + late).float() * self.step_dt

        at_last = self._wp_idx >= (self._num_wp - 1)
        self._task_done = self._reached & at_last
        advance = self._reached & ~at_last
        self._wp_idx = torch.where(advance, self._wp_idx + 1, self._wp_idx)
        # 到達下一個路徑點就重新計時
        self._wp_step_counter = torch.where(
            advance, torch.zeros_like(self._wp_step_counter), self._wp_step_counter
        )

        # 超時：在該路徑點各自的時間預算內沒到達（每個點的預算不同，依當前目標點索引取值）
        self._wp_timeout = self._wp_step_counter >= self._wp_timeout_steps[self._wp_idx]
        # 失敗：手臂力量不足撐不住 / 嚴重脫離軌跡，或是超時
        drift_failed = self._dist > self.cfg.max_target_dist
        self._task_failed = drift_failed | self._wp_timeout

        # 下一步的進度基準（若路徑點已切換，改用到新目標的距離）
        new_target = self._waypoints[self._wp_idx]
        self._prev_dist = torch.norm(ee_pos_b - new_target[:, 0:3], dim=-1)

        # 功率（電力）: P = Σ|τ·ω| + heat_coeff·Στ²
        tau = self._robot.data.applied_torque[:, self._arm_joint_ids]
        omega = self._robot.data.joint_vel[:, self._arm_joint_ids]
        p_mech = torch.sum(torch.abs(tau * omega), dim=-1)
        p_heat = self.cfg.heat_coeff * torch.sum(tau * tau, dim=-1)
        self._power_W = p_mech + p_heat

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_task_state()
        terminated = self._task_done | self._task_failed
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        energy_J = self._power_W * self.step_dt  # 這一步消耗的電能（焦耳）
        limit_norm = ((self._effort_limits - self._effort_min) /
                      (self._effort_max - self._effort_min)).mean(dim=-1)
        action_rate = torch.sum((self.actions - self._prev_actions) ** 2, dim=-1)

        r_progress = cfg.rew_progress_weight * self._progress
        r_waypoint = cfg.rew_waypoint_bonus * self._reached.float()
        r_success = cfg.rew_success_bonus * self._task_done.float()
        # 抵達時間偏離窗口 [MIN, MAX] 的懲罰（∝ 偏離秒數，只在抵達路徑點那一步非零）
        p_window = -cfg.pen_window_weight * self._window_pen
        p_dist = -cfg.pen_dist_weight * self._dist
        p_energy = -cfg.pen_energy_weight * energy_J
        p_limit = -cfg.pen_effort_limit_weight * limit_norm
        p_rate = -cfg.pen_action_rate_weight * action_rate
        # 超時跟一般失敗（脫離軌跡）是互斥的懲罰，避免同一步被扣兩次
        p_timeout = -cfg.pen_timeout * self._wp_timeout.float()
        p_fail = -cfg.pen_fail * (self._task_failed & ~self._wp_timeout).float()

        reward = (
            r_progress + r_waypoint + r_success + p_window
            + p_dist + p_energy + p_limit + p_rate + p_fail + p_timeout
        )

        # ---- 統計 ----
        self._ep_sums["progress"] += r_progress
        self._ep_sums["waypoint_bonus"] += r_waypoint
        self._ep_sums["success_bonus"] += r_success
        self._ep_sums["window_pen"] += p_window
        self._ep_sums["dist_pen"] += p_dist
        self._ep_sums["energy_pen"] += p_energy
        self._ep_sums["limit_pen"] += p_limit
        self._ep_sums["action_rate_pen"] += p_rate
        self._ep_sums["timeout_pen"] += p_timeout
        self._ep_sums["energy_J"] += energy_J
        self._ep_limit_sums += self._effort_limits
        return reward

    # ------------------------------------------------------------------
    # 觀測
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        ee_pos_b, _ = self._ee_pose_b()
        target = self._waypoints[self._wp_idx]
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        default_pos = self._robot.data.default_joint_pos[:, self._arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]

        obs = torch.cat(
            (
                joint_pos - default_pos,                        # 7
                joint_vel * 0.1,                                # 7
                self.actions,                                   # 7  當前功率上限（正規化）
                ee_pos_b,                                       # 3
                target[:, 0:3] - ee_pos_b,                      # 3  位置誤差向量
                target[:, 0:3],                                 # 3  當前目標點
                (self._wp_idx.float() / self._num_wp).unsqueeze(-1),  # 1  任務進度
                torch.full((self.num_envs, 1), self.cfg.payload_mass, device=self.device),  # 1
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # ---- 寫入 episode 統計到 log ----
        if len(env_ids) > 0:
            extras = dict()
            ep_len = (self.episode_length_buf[env_ids].float() + 1.0).clamp(min=1.0)
            for key, buf in self._ep_sums.items():
                prefix = "Metrics" if key == "energy_J" else "Episode_Reward"
                extras[f"{prefix}/{key}"] = torch.mean(buf[env_ids])
                buf[env_ids] = 0.0
            # 各關節在這個 episode 內的平均力矩上限 -> 「最佳輸出功率」的答案
            mean_limits = self._ep_limit_sums[env_ids] / ep_len.unsqueeze(-1)
            for j in range(7):
                extras[f"EffortLimit_Nm/panda_joint{j + 1}"] = torch.mean(mean_limits[:, j])
            self._ep_limit_sums[env_ids] = 0.0
            extras["Metrics/waypoints_reached"] = torch.mean(
                self._wp_idx[env_ids].float() + self._task_done[env_ids].float()
            )
            extras["Metrics/success_rate"] = torch.mean(self._task_done[env_ids].float())
            extras["Metrics/mean_power_W"] = torch.mean(self._power_W[env_ids])
            extras["Metrics/timeout_rate"] = torch.mean(self._wp_timeout[env_ids].float())
            self.extras["log"] = extras

        super()._reset_idx(env_ids)

        # ---- 重置機器人狀態（預設關節角 + 小幅隨機）----
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        noise = sample_uniform(
            -self.cfg.reset_joint_noise, self.cfg.reset_joint_noise,
            (len(env_ids), len(self._arm_joint_ids)), self.device,
        )
        joint_pos[:, self._arm_joint_ids] += noise
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_root_pose_to_sim(self._robot.data.default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(
            self._robot.data.default_root_state[env_ids, 7:], env_ids
        )
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ---- 重置任務狀態 ----
        self._wp_idx[env_ids] = 0
        self._wp_step_counter[env_ids] = 0
        self._wp_timeout[env_ids] = False
        self._prev_dist[env_ids] = -1.0  # 標記為剛重置，下一步進度為 0
        self._task_done[env_ids] = False
        self._task_failed[env_ids] = False
        self.actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._joint_pos_des[env_ids] = joint_pos[:, self._arm_joint_ids]
