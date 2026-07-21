# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka 最小功率任務設定檔（manager-based workflow）。

=======================================================================
使用者主要修改的區域集中在這個檔案最上方（與 Direct 版一致）：

1. ``WAYPOINTS``           : 末端要依序走過的位置（機器人基座座標系）
2. ``WAYPOINT_TIMEOUT_S``  : 每個路徑點各自的抵達時間預算（秒）
3. ``PAYLOAD_MASS``        : 末端模擬物品的重量 (kg)
4. ``EFFORT_LIMIT_MIN/MAX``: 各關節力矩上限（輸出功率）的搜尋範圍

獎勵權重放在 ``RewardsCfg``（每個 RewTerm 的 weight），其餘任務參數在
``FrankaPowerEnvCfg`` 內，皆有註解。
=======================================================================
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from . import mdp

# 重用 Direct 版定義好的機器人設定（官方 Franka，手臂換顯式 IdealPDActuator）
from ...direct.franka_power.franka_power_env_cfg import FRANKA_POWER_ROBOT_CFG  # isort: skip


# =====================================================================
# ★★★ 使用者參數（自成一格，可獨立於 Direct 版編輯）★★★
# =====================================================================
WAYPOINTS = [
    [0.50,  0.00, 0.35],
    [0.50,  0.30, 0.55],
    [0.45, -0.30, 0.55],
    [0.55,  0.00, 0.70],
]
WAYPOINT_TIMEOUT_S = [3.0, 3.0, 3.0, 3.0]      # 與 WAYPOINTS 一一對應，長度須相同
DEFAULT_EE_QUAT = (0.0, 1.0, 0.0, 0.0)          # 未指定姿態時的預設末端姿態 (w,x,y,z)
PAYLOAD_MASS = 1.0                              # 末端模擬物品重量 (kg)
EFFORT_LIMIT_MIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
EFFORT_LIMIT_MAX = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]  # 官方額定值作為上限


##
# 場景
##
@configclass
class FrankaPowerSceneCfg(InteractiveSceneCfg):
    robot = FRANKA_POWER_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# 動作：7 維 = 各關節力矩上限（正規化），軌跡由差分 IK 產生
##
@configclass
class ActionsCfg:
    effort_limit_ik: mdp.EffortLimitIKActionCfg = mdp.EffortLimitIKActionCfg(asset_name="robot")


##
# 觀測
##
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_state = ObsTerm(func=mdp.robot_state)      # 21
        ee_and_target = ObsTerm(func=mdp.ee_and_target)  # 9
        task_progress = ObsTerm(func=mdp.task_progress)  # 2

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# 獎勵（權重 = Direct 版數值；RewardManager 會再乘 dt，各項相對平衡不變）
##
@configclass
class RewardsCfg:
    progress = RewTerm(func=mdp.progress, weight=50.0)
    waypoint_bonus = RewTerm(func=mdp.waypoint_bonus, weight=10.0)
    success_bonus = RewTerm(func=mdp.success_bonus, weight=50.0)
    speed_bonus = RewTerm(func=mdp.speed_bonus, weight=100.0)
    dist_pen = RewTerm(func=mdp.distance_penalty, weight=-0.5)
    energy_pen = RewTerm(func=mdp.energy_penalty, weight=-0.05)
    limit_pen = RewTerm(func=mdp.effort_limit_penalty, weight=-0.05)
    action_rate_pen = RewTerm(func=mdp.action_rate_penalty, weight=-0.01)
    timeout_pen = RewTerm(func=mdp.timeout_penalty, weight=-100.0)
    fail_pen = RewTerm(func=mdp.failure_penalty, weight=-20.0)


##
# 終止
##
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # episode 時間上限（truncation）
    success = DoneTerm(func=mdp.task_success)              # 走完全程
    failure = DoneTerm(func=mdp.task_failure)              # 脫離軌跡 / 超時


##
# 事件
##
@configclass
class EventCfg:
    apply_payload = EventTerm(func=mdp.apply_payload, mode="startup")
    reset_task = EventTerm(func=mdp.reset_power_task, mode="reset")


##
# 環境總設定
##
@configclass
class FrankaPowerEnvCfg(ManagerBasedRLEnvCfg):
    # ---- 使用者任務參數（由 PowerTaskEngine 透過 env.cfg 讀取）----
    waypoints: list = WAYPOINTS
    waypoint_timeout_s: list = WAYPOINT_TIMEOUT_S
    default_ee_quat: tuple = DEFAULT_EE_QUAT
    payload_mass: float = PAYLOAD_MASS
    effort_limit_min: list = EFFORT_LIMIT_MIN
    effort_limit_max: list = EFFORT_LIMIT_MAX

    waypoint_tolerance: float = 0.04   # 末端距離路徑點多近算「到達」(m)
    max_target_dist: float = 0.9       # 末端偏離目標超過此距離 -> 失敗 (m)
    reset_joint_noise: float = 0.05    # 重置時關節角度隨機擾動 (rad)
    ik_method: str = "dls"             # 差分 IK 方法
    heat_coeff: float = 0.005          # 銅損係數 W/(Nm)²，P = Σ|τω| + heat_coeff·Στ²

    # ---- managers ----
    scene: FrankaPowerSceneCfg = FrankaPowerSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 12.0
        self.sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=self.decimation)
        self.viewer.eye = (2.5, 2.5, 2.0)
