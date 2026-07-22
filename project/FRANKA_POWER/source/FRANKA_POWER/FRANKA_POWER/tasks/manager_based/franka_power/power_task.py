# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""共享任務引擎：把 Direct 版 FrankaPowerEnv 裡的有狀態邏輯集中成一個物件。

manager-based 工作流沒有天然的「每步 post-physics hook」，也沒有一個地方能放
跨 observation / reward / termination 共用的狀態。這個引擎就是那個共用狀態：

* 由自訂的 ActionTerm（``EffortLimitIKAction``）在建構時建立，並掛到
  ``env.power_task`` 供所有 mdp 函式取用。
* ``begin_step()`` 在 ActionManager.process_action（pre-physics）呼叫：把 Agent
  動作轉成各關節力矩上限、寫進顯式致動器、跑差分 IK 解出關節位置目標。
* ``apply()`` 在每個 decimation 子步（pre-physics）呼叫：把關節位置目標下給機器人。
* ``update()`` 在 reward / termination 函式（post-physics）呼叫，用 step-token
  確保「每個 policy step 只真正計算一次」：算距離、推進路徑點、判定超時/失敗、算功率。
* observation 函式改讀「即時狀態」（不經過 update 快取），這樣 reset 之後回傳的
  觀測才會是新狀態，而不是上一個 episode 的殘值。
"""

from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms


class PowerTaskEngine:
    """FRANKA_POWER 任務的共用狀態與每步計算。"""

    def __init__(self, env, asset_name: str = "robot"):
        self._env = env
        self.cfg = env.cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self.step_dt = env.step_dt

        self._robot = env.scene[asset_name]

        # ---- 關節 / body 索引 ----
        self._arm_joint_ids, _ = self._robot.find_joints("panda_joint[1-7]")
        self._finger_joint_ids, _ = self._robot.find_joints("panda_finger_joint.*")
        self._hand_body_idx = self._robot.find_bodies("panda_hand")[0][0]
        # 固定基座機器人的 Jacobian 不含 root body，索引要 -1
        self._ee_jacobi_idx = self._hand_body_idx - 1
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
        self.waypoints = torch.tensor(wp_list, dtype=torch.float, device=self.device)
        self.num_wp = self.waypoints.shape[0]

        # ---- 每個路徑點各自的時間預算 ----
        if len(self.cfg.waypoint_timeout_s) != self.num_wp:
            raise ValueError(
                f"waypoint_timeout_s 長度 ({len(self.cfg.waypoint_timeout_s)}) 必須跟 "
                f"waypoints 數量 ({self.num_wp}) 一致，兩者要一一對應。"
            )
        self._wp_timeout_steps = torch.tensor(
            [max(1, int(round(t / self.step_dt))) for t in self.cfg.waypoint_timeout_s],
            dtype=torch.long, device=self.device,
        )
        # 抵達時間窗下界（每個路徑點；0 表示沒有下界，不罰早到）
        if len(self.cfg.waypoint_time_min_s) != self.num_wp:
            raise ValueError(
                f"waypoint_time_min_s 長度 ({len(self.cfg.waypoint_time_min_s)}) 必須跟 "
                f"waypoints 數量 ({self.num_wp}) 一致，兩者要一一對應。"
            )
        self._wp_min_steps = torch.tensor(
            [max(0, int(round(t / self.step_dt))) for t in self.cfg.waypoint_time_min_s],
            dtype=torch.long, device=self.device,
        )

        # ---- 差分 IK ----
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method=self.cfg.ik_method
        )
        self._diff_ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # ---- 力矩上限搜尋範圍 ----
        self._effort_min = torch.tensor(self.cfg.effort_limit_min, device=self.device)
        self._effort_max = torch.tensor(self.cfg.effort_limit_max, device=self.device)

        # ---- 動作 / 效果緩衝 ----
        self.actions_norm = torch.zeros(self.num_envs, 7, device=self.device)
        self._prev_actions = torch.zeros_like(self.actions_norm)
        self.effort_limits = self._effort_max.unsqueeze(0).repeat(self.num_envs, 1)
        self._joint_pos_des = self._robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

        # ---- 任務狀態 ----
        self.wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._wp_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_dist = torch.full((self.num_envs,), -1.0, device=self.device)
        self.dist = torch.zeros(self.num_envs, device=self.device)
        self.progress = torch.zeros(self.num_envs, device=self.device)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.power_W = torch.zeros(self.num_envs, device=self.device)
        self.energy_J = torch.zeros(self.num_envs, device=self.device)
        self.limit_norm = torch.zeros(self.num_envs, device=self.device)
        self.action_rate = torch.zeros(self.num_envs, device=self.device)
        self.window_pen = torch.zeros(self.num_envs, device=self.device)  # 抵達時間偏離窗口的秒數

        # ---- episode 統計（給 _reset_idx 記 log 用）----
        self._ep_limit_sums = torch.zeros(self.num_envs, 7, device=self.device)
        self._ep_energy_sum = torch.zeros(self.num_envs, device=self.device)

        # ---- step-token：確保 update() 每個 policy step 只算一次 ----
        self._step_id = 0
        self._computed_step = -1

        self._payload_applied = False

        print(
            f"[PowerTaskEngine] 路徑點數量: {self.num_wp}, 末端負載: {self.cfg.payload_mass} kg, "
            f"各點時間預算(s): {self.cfg.waypoint_timeout_s} -> steps: {self._wp_timeout_steps.tolist()}"
        )

    # ------------------------------------------------------------------
    # 幾何小工具
    # ------------------------------------------------------------------

    def ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """末端（panda_hand）在基座座標系的位置與姿態（即時計算）。"""
        ee_pose_w = self._robot.data.body_pose_w[:, self._hand_body_idx]
        root_pose_w = self._robot.data.root_pose_w
        return subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

    def current_target(self) -> torch.Tensor:
        """各環境目前要追蹤的路徑點 (num_envs, 7)。"""
        return self.waypoints[self.wp_idx]

    # ------------------------------------------------------------------
    # 末端負載
    # ------------------------------------------------------------------

    def apply_payload(self):
        """把負載質量加到 panda_hand，慣量依質量比例放大（僅執行一次）。"""
        if self._payload_applied or self.cfg.payload_mass <= 0.0:
            return
        masses = self._robot.root_physx_view.get_masses().clone()
        inertias = self._robot.root_physx_view.get_inertias().clone()
        hand_mass = masses[:, self._hand_body_idx].clone()
        ratio = (hand_mass + self.cfg.payload_mass) / hand_mass
        masses[:, self._hand_body_idx] += self.cfg.payload_mass
        inertias[:, self._hand_body_idx, :] *= ratio.unsqueeze(-1)
        all_ids = torch.arange(self.num_envs)
        self._robot.root_physx_view.set_masses(masses, all_ids)
        self._robot.root_physx_view.set_inertias(inertias, all_ids)
        self._payload_applied = True

    # ------------------------------------------------------------------
    # pre-physics：動作 -> 力矩上限 + IK
    # ------------------------------------------------------------------

    def begin_step(self, actions: torch.Tensor):
        """ActionManager.process_action（每個 policy step 開頭）呼叫。"""
        self._prev_actions = self.actions_norm.clone()
        self.actions_norm = actions.clamp(-1.0, 1.0).clone()
        # 正規化動作 -> 各關節力矩上限（輸出功率）
        self.effort_limits = self._effort_min + 0.5 * (self.actions_norm + 1.0) * (
            self._effort_max - self._effort_min
        )
        self._arm_actuator.effort_limit[:] = self.effort_limits

        # 差分 IK：朝當前路徑點解出關節位置目標
        target_pose = self.current_target()
        self._diff_ik.set_command(target_pose)
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, :
        ][:, :, self._arm_joint_ids]
        ee_pos_b, ee_quat_b = self.ee_pose_b()
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        self._joint_pos_des = self._diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # 新的一個 policy step 開始了，標記需要重算 post-physics 狀態
        self._step_id += 1

    def apply(self):
        """每個 decimation 子步呼叫：把關節位置目標下給機器人。"""
        self._robot.set_joint_position_target(self._joint_pos_des, joint_ids=self._arm_joint_ids)
        self._robot.set_joint_position_target(
            self._robot.data.default_joint_pos[:, self._finger_joint_ids],
            joint_ids=self._finger_joint_ids,
        )

    # ------------------------------------------------------------------
    # post-physics：任務狀態（每步只算一次）
    # ------------------------------------------------------------------

    def update(self):
        """reward / termination 函式呼叫；同一 policy step 內只真正計算一次。"""
        if self._computed_step == self._step_id:
            return
        self._computed_step = self._step_id

        self._wp_step_counter += 1

        ee_pos_b, _ = self.ee_pose_b()
        target = self.current_target()
        self.dist = torch.norm(ee_pos_b - target[:, 0:3], dim=-1)

        # 進度 = 上一步距離 - 這一步距離（剛重置的環境進度為 0）
        fresh = self._prev_dist < 0.0
        self.progress = torch.where(fresh, torch.zeros_like(self.dist), self._prev_dist - self.dist)

        # 到達判定與路徑點推進
        self.reached = self.dist < self.cfg.waypoint_tolerance

        # 抵達時間窗懲罰：用「推進前」的 wp_idx 與計數器，算這次抵達偏離 [MIN, MAX] 幾秒
        min_steps = self._wp_min_steps[self.wp_idx]
        max_steps = self._wp_timeout_steps[self.wp_idx]
        early = (min_steps - self._wp_step_counter).clamp(min=0)
        late = (self._wp_step_counter - max_steps).clamp(min=0)
        self.window_pen = self.reached.float() * (early + late).float() * self.step_dt

        at_last = self.wp_idx >= (self.num_wp - 1)
        self.task_done = self.reached & at_last
        advance = self.reached & ~at_last
        self.wp_idx = torch.where(advance, self.wp_idx + 1, self.wp_idx)
        self._wp_step_counter = torch.where(
            advance, torch.zeros_like(self._wp_step_counter), self._wp_step_counter
        )

        # 超時：在該路徑點各自的時間預算內沒到達
        self.timeout = self._wp_step_counter >= self._wp_timeout_steps[self.wp_idx]
        drift_failed = self.dist > self.cfg.max_target_dist
        self.task_failed = drift_failed | self.timeout

        # 下一步的進度基準（路徑點切換後改用到新目標的距離）
        new_target = self.current_target()
        self._prev_dist = torch.norm(ee_pos_b - new_target[:, 0:3], dim=-1)

        # 功率（電力）: P = Σ|τ·ω| + heat_coeff·Στ²
        tau = self._robot.data.applied_torque[:, self._arm_joint_ids]
        omega = self._robot.data.joint_vel[:, self._arm_joint_ids]
        p_mech = torch.sum(torch.abs(tau * omega), dim=-1)
        p_heat = self.cfg.heat_coeff * torch.sum(tau * tau, dim=-1)
        self.power_W = p_mech + p_heat
        self.energy_J = self.power_W * self.step_dt

        # 供 reward 使用的衍生量
        self.limit_norm = ((self.effort_limits - self._effort_min) /
                           (self._effort_max - self._effort_min)).mean(dim=-1)
        self.action_rate = torch.sum((self.actions_norm - self._prev_actions) ** 2, dim=-1)

        # episode 統計累積
        self._ep_limit_sums += self.effort_limits
        self._ep_energy_sum += self.energy_J

    # ------------------------------------------------------------------
    # 記錄與重置
    # ------------------------------------------------------------------

    def episode_metrics(self, env_ids: torch.Tensor) -> dict:
        """回傳這批環境「剛結束的 episode」自訂 metric（在 reset 之前呼叫）。"""
        ep_len = self._env.episode_length_buf[env_ids].float().clamp(min=1.0)
        mean_limits = self._ep_limit_sums[env_ids] / ep_len.unsqueeze(-1)
        log = {f"EffortLimit_Nm/panda_joint{j + 1}": torch.mean(mean_limits[:, j]) for j in range(7)}
        log["Metrics/energy_J"] = torch.mean(self._ep_energy_sum[env_ids])
        log["Metrics/waypoints_reached"] = torch.mean(
            self.wp_idx[env_ids].float() + self.task_done[env_ids].float()
        )
        log["Metrics/success_rate"] = torch.mean(self.task_done[env_ids].float())
        log["Metrics/mean_power_W"] = torch.mean(self.power_W[env_ids])
        log["Metrics/timeout_rate"] = torch.mean(self.timeout[env_ids].float())
        return log

    def reset(self, env_ids: torch.Tensor | None = None):
        """重置機器人狀態 + 任務緩衝（由 reset 事件呼叫）。"""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device).long()
        if env_ids.numel() == 0:
            return

        # 機器人狀態（預設關節角 + 手臂小幅隨機）
        from isaaclab.utils.math import sample_uniform

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        noise = sample_uniform(
            -self.cfg.reset_joint_noise, self.cfg.reset_joint_noise,
            (len(env_ids), len(self._arm_joint_ids)), self.device,
        )
        joint_pos[:, self._arm_joint_ids] += noise
        joint_vel = torch.zeros_like(joint_pos)
        default_root = self._robot.data.default_root_state[env_ids]
        self._robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 任務緩衝
        self.wp_idx[env_ids] = 0
        self._wp_step_counter[env_ids] = 0
        self.timeout[env_ids] = False
        self._prev_dist[env_ids] = -1.0
        self.task_done[env_ids] = False
        self.task_failed[env_ids] = False
        self.reached[env_ids] = False
        self.actions_norm[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._joint_pos_des[env_ids] = joint_pos[:, self._arm_joint_ids]
        self._ep_limit_sums[env_ids] = 0.0
        self._ep_energy_sum[env_ids] = 0.0
