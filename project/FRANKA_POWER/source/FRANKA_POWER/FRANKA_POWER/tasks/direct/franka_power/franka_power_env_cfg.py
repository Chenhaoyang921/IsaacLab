# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka 最小功率任務設定檔。

=======================================================================
使用者主要修改的區域都集中在這個檔案最上方：

1. ``WAYPOINTS``     : 機械手末端要依序走過的位置（機器人基座座標系）
2. ``PAYLOAD_MASS``  : 末端模擬物品的重量 (kg)，慣量會等比例增加
3. ``EFFORT_LIMIT_MAX`` : 各關節允許的力矩上限搜尋範圍（Agent 在
   [EFFORT_LIMIT_MIN, EFFORT_LIMIT_MAX] 之間選擇每個關節的輸出功率）
=======================================================================
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

##
# 官方 Franka Panda 手臂（來自 isaaclab_assets）
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


# =====================================================================
# ★★★ 1. 路徑點：機械手末端要依序經過的位置 ★★★
#
# 每一列可以是：
#   [x, y, z]                      -> 只指定位置，姿態使用預設（夾爪朝下）
#   [x, y, z, qw, qx, qy, qz]      -> 完整指定位置 + 姿態（四元數 w,x,y,z）
#
# 座標系：機器人基座（底座在原點，x 向前、z 向上），單位：公尺。
# Franka 的工作範圍大約在半徑 0.85m 內、高度 0.1~1.1m。
# =====================================================================
WAYPOINTS = [
    [0.50,  0.00, 0.35],
    [0.50,  0.30, 0.55],
    [0.45, -0.30, 0.55],
    [0.55,  0.00, 0.70],
]

# 每個路徑點的「抵達時間窗」下界與上界 (秒)，都跟 WAYPOINTS 一一對應、長度相同。
# 從「上一個點到達」或「episode 開始」起算，Agent 應在 [MIN, MAX] 之間抵達：
#   * 早於 MIN 抵達（太快/太衝）        -> 扣 pen_window 分，∝ 早到的秒數
#   * 晚於 MAX 抵達（= 超過 TIMEOUT）    -> 視為失敗，扣 pen_timeout 分並結束該 episode
# 也就是說 WAYPOINT_TIMEOUT_S 同時是「時間窗上界」與「硬性超時失敗線」。
WAYPOINT_TIME_MIN_S = [
    1.0,
    1.0,
    1.0,
    1.0,
]
WAYPOINT_TIMEOUT_S = [
    3.0,
    3.0,
    3.0,
    3.0,
]

# 未指定姿態時使用的預設末端姿態（夾爪朝下），四元數 (w, x, y, z) ，預設:(0.0, 1.0, 0.0, 0.0)
DEFAULT_EE_QUAT = (0.0, 1.0, 0.0, 0.0)

# =====================================================================
# ★★★ 2. 末端模擬物品的重量 ★★★
#
# 單位 kg，會加到 panda_hand 這個 body 上，慣量依質量比例放大。
# Franka 官方額定荷重為 3 kg。
# =====================================================================
PAYLOAD_MASS = 1.0
    
# =====================================================================
# ★★★ 3. 關節輸出功率（力矩上限）搜尋範圍 ★★★
#
# Agent 的動作空間就是每個關節的力矩上限（Nm），
# 每一步在 [MIN, MAX] 之間連續選擇。
# 官方額定值：joint1-4 = 87 Nm，joint5-7 = 12 Nm，這裡作為搜尋上限。
# =====================================================================
EFFORT_LIMIT_MIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
EFFORT_LIMIT_MAX = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]


##
# 機器人設定：官方 Franka，但手臂 7 個關節換成顯式 IdealPDActuator。
# 顯式致動器讓我們可以：
#  (a) 每一步動態改寫 effort_limit（Agent 的動作）
#  (b) 從 applied_torque 讀到被截斷後「實際輸出」的力矩 -> 精確計算功率
##
FRANKA_POWER_ROBOT_CFG: ArticulationCfg = FRANKA_PANDA_CFG.replace(
    prim_path="/World/envs/env_.*/Robot"
)
FRANKA_POWER_ROBOT_CFG.spawn.rigid_props.disable_gravity = False
FRANKA_POWER_ROBOT_CFG.actuators = {
    "panda_arm": IdealPDActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        # 初始力矩上限（訓練時每步都會被 Agent 的動作覆寫）
        effort_limit={"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0},
        velocity_limit={"panda_joint[1-4]": 2.175, "panda_joint[5-7]": 2.61},
        # IK 追蹤用的 PD 增益（與官方 FRANKA_PANDA_HIGH_PD_CFG 相同）
        stiffness=400.0,
        damping=80.0,
    ),
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"],
        effort_limit_sim=200.0,
        stiffness=2e3,
        damping=1e2,
    ),
}


@configclass
class FrankaPowerEnvCfg(DirectRLEnvCfg):
    """最小功率走點任務。

    Agent 動作 = 7 個關節的力矩上限（正規化 [-1,1] -> [MIN, MAX] Nm）。
    軌跡由差分 IK 自動產生，Agent 只負責決定「給每個關節多少力」。
    """

    # ---- 基本 RL 設定 ----
    episode_length_s = 12.0     # 單次 episode 秒數（要足夠走完所有路徑點）
    decimation = 2              # policy 頻率 = 1/(dt*decimation) = 60 Hz
    action_space = 7            # 7 個關節的力矩上限
    observation_space = 32
    state_space = 0

    # ---- 模擬 ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=True
    )

    # ---- 機器人 ----
    robot: ArticulationCfg = FRANKA_POWER_ROBOT_CFG

    # ---- 任務參數（從模組頂部帶入，也可直接在此覆寫）----
    waypoints: list = WAYPOINTS
    waypoint_time_min_s: list = WAYPOINT_TIME_MIN_S  # 每個路徑點的抵達時間窗下界 (秒)，須與 waypoints 等長
    waypoint_timeout_s: list = WAYPOINT_TIMEOUT_S    # 抵達時間窗上界＝硬性超時失敗線 (秒)，須與 waypoints 等長
    default_ee_quat: tuple = DEFAULT_EE_QUAT
    payload_mass: float = PAYLOAD_MASS
    effort_limit_min: list = EFFORT_LIMIT_MIN
    effort_limit_max: list = EFFORT_LIMIT_MAX

    waypoint_tolerance: float = 0.04   # 末端距離路徑點多近算「到達」(m)
    max_target_dist: float = 0.9       # 末端偏離目標超過此距離 -> 任務失敗 (m)
    reset_joint_noise: float = 0.05    # 重置時關節角度隨機擾動 (rad)

    # ---- IK 控制器 ----
    ik_method: str = "dls"             # damped least squares

    # ---- 功率（電力）模型 ----
    # P_elec = Σ|τ·ω|（機械功率） + heat_coeff · Στ²（銅損，馬達持力發熱）
    # heat_coeff ≈ R/(kt)²，預設 0.005 W/(Nm)²
    heat_coeff: float = 0.005

    # ---- 獎勵權重 ----
    rew_progress_weight: float = 50.0    # 朝目標點前進的距離 (m) × 此權重
    rew_waypoint_bonus: float = 10.0     # 每到達一個路徑點
    rew_success_bonus: float = 50.0      # 走完全部路徑點
    # 抵達路徑點時，若落在時間窗 [MIN, MAX] 之外，扣分 ∝ 偏離秒數（每偏離 1 秒扣此權重）。
    # 預設情況下 MAX = 超時失敗線，所以主要作用在「早到」那一側，逼 Agent 不要太衝、
    # 在規定時間才抵達；晚到那一側由 pen_timeout（硬性失敗）負責。
    pen_window_weight: float = 30.0
    pen_dist_weight: float = 0.5         # 每步對「距目標距離」的持續懲罰
    pen_energy_weight: float = 0.05      # 每消耗 1 焦耳的懲罰（核心：最小電力）
    pen_effort_limit_weight: float = 0.05  # 對「開太大功率上限」本身的小額懲罰
    pen_action_rate_weight: float = 0.01   # 動作變化率懲罰（讓功率設定平滑）
    pen_fail: float = 20.0               # 任務失敗（手臂撐不住而脫離軌跡，非超時）
    pen_timeout: float = 100.0           # 超過 waypoint_timeout_s 還沒到下一個點的懲罰
