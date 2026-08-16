# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg

from FLOWER_INSERT.tasks.manager_based.cabinet import mdp       # 導入馬可夫決策過程相關的邏輯函數

##
# Pre-defined configs
##
##
# 預定義配置
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

# 建立一個較小的座標軸標記，用於視覺化偵錯
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

# 資產目錄（相對於本檔案），避免寫死特定電腦的絕對路徑
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


##
# Scene definition
##
##
# 場景定義 (Scene Definition)
##

@configclass
class FlowerSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    # 機器人設定：目前設為 MISSING，會在具體的 Agent 配置中填入 (例如 Franka 或 EPSON)
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    # 末端執行器框架：同上，定義機器人的手部位置
    ee_frame: FrameTransformerCfg = MISSING

    # 花朵資產配置（靜態物件，使用 AssetBaseCfg）
    flower = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Flower",
        spawn=sim_utils.UsdFileCfg(
            # 花朵 USD 模型路徑
            usd_path=os.path.join(ASSETS_DIR, "Flower_02.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            # 不設 mass_props，改由 USD 內既有的質量/慣量定義（Isaac Lab 不覆寫）
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0, 0.025),  # 花朵放置的位置
            rot=(1.0, 0.0, 0.0, 0.0),  # 繞 Z 軸再轉 180° → X 軸朝前（+X）
        ),
    )

    # Frame definitions for the cabinet.
    # 定義花朵的座標轉換器，用於追蹤「花朵中心」的位置
    flower_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Flower",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/FlowerFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Flower",
                name="flower_center",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.01),
                    rot=(1.0, 0.0, 0.0, 0.0),  # align with end-effector frame
                ),
            ),
        ],
    )

    # 瓶子資產配置（目標放置位置）
    bottle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "BOTTLE.usd"),
            scale=(0.003, 0.002, 0.003),   # mm → m，放大 2 倍；x/z 再額外放大 1.5 倍
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.3, 0.0),   # 初始放在約 1m 外，Stage 3 開始時瞬移到花朵位置
            rot=(0.7071, 0.7071, 0.0, 0.0),  # 繞 X 軸旋轉 90°（Y 方向 → Z 方向）
        ),
    )

    # 瓶子座標轉換器，用於追蹤瓶子中心位置
    bottle_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/BottleFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                name="bottle_top",           # index 0
                offset=OffsetCfg(
                    pos=(0.0, 0.5, 0.0),
                    rot=(0.7071, 0.7071, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                name="bottle_inside",        # index 1
                offset=OffsetCfg(
                    pos=(0.0, 0.248, 0.0),
                    rot=(0.7071, 0.7071, 0.0, 0.0),
                ),
            ),
        ],
    )

    # 夾爪接觸感測器：偵測兩指接觸力（index 0 = leftfinger, index 1 = rightfinger）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    # plane
    # 地板設定
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,     # 碰撞分組
    )

    # 每個環境的工作檯面（薄方塊，頂面與地面齊平 z=0）
    # 使用 per-env prim path，可作為 contact sensor 的 filter 目標
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.01, -0.025)),  # 頂面在 z=0
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.2, 0.05),  # 長 × 寬 × 厚（公尺）
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.5, 0.4),  # 木頭色
                roughness=0.8,
            ),
        ),
    )

    # lights
    # 燈光設定
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings  (馬可夫決策過程)
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    """定義 AI 可以執行的動作。"""
    arm_action: mdp.JointPositionActionCfg = MISSING                # 手臂關節位置控制
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING      # 夾爪開合控制 (0 或 1)


@configclass
class ObservationsCfg:
    """定義 AI 看到的觀察值 (眼睛)。"""
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        """策略網路觀察組。"""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)             # 機器人自身關節相對位置
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)             # 機器人自身關節速度

        # 觀察花朵位置（相對於 EE）
        flower_pos = ObsTerm(func=mdp.flower_pos)

        # 手部與花朵的距離（第一階段用）
        rel_ee_flower_distance = ObsTerm(func=mdp.rel_ee_flower_distance)

        # 花朵到瓶口的距離（第三階段用）
        rel_flower_bottle_distance = ObsTerm(func=mdp.rel_flower_bottle_distance)

        # 任務階段標籤：[0,0]=第一階段，[1,0]=第三階段
        stage_tag = ObsTerm(func=mdp.stage_tag)

        actions = ObsTerm(func=mdp.last_action) # 上一步執行的動作（幫助動作連續性）

        def __post_init__(self):
            self.enable_corruption = True       # 開啟觀察值雜訊（增加穩定性）
            self.concatenate_terms = True       # 將所有觀察值合併為一個長向量

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    """配置環境隨機化事件（Domain Randomization）。"""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup", # 僅在啟動時執行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),        # 恢復係數（彈跳力）
            "num_buckets": 16,
        },
    )



    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")    # 每次回合重置場景

    # 每次重置時給關節位置加一點點隨機偏移
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )





# ── 各階段密集獎勵權重（唯一來源，complete_bonus 由這些加總推導）──────────
# s2_release 不計入：weight 在這裡是 MISSING，實際由 franka 的 joint_pos_env_cfg.py
# 覆寫為 0.0，加不加都不影響總和。
_S0_DENSE_WEIGHTS = (0.2, 1.0, 10.0)       # approach_flower, align_flower, multi_lift
_S1_DENSE_WEIGHTS = (50.0, 50.0)           # approach_bottle, align_flower_up
_S2_DENSE_WEIGHTS = (120.0,)               # approach_inside

S0_COMPLETE_BONUS_WEIGHT = sum(_S0_DENSE_WEIGHTS)
S1_COMPLETE_BONUS_WEIGHT = sum(_S1_DENSE_WEIGHTS)
S2_COMPLETE_BONUS_WEIGHT = sum(_S2_DENSE_WEIGHTS)
ALL_COMPLETE_BONUS_WEIGHT = S0_COMPLETE_BONUS_WEIGHT + S1_COMPLETE_BONUS_WEIGHT + S2_COMPLETE_BONUS_WEIGHT


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    """獎勵機制定義。"""
    # ── Stage 0：接近、對齊、抓取、舉起花朵 ────────────────────────────
    s0_approach_flower = RewTerm(func=mdp.s0_approach_flower, weight=_S0_DENSE_WEIGHTS[0], params={"threshold": 0.2})
    s0_align_flower    = RewTerm(func=mdp.s0_align_flower, weight=_S0_DENSE_WEIGHTS[1])

    s0_multi_lift          = RewTerm(func=mdp.s0_multi_lift, weight=_S0_DENSE_WEIGHTS[2])
    # dead_penalty = -λ × cum_end(400+300+200=900)，λ=1
    s0_dead_penalty        = RewTerm(func=mdp.dead_penalty, weight=-900.0)
    # complete_bonus 權重 = S0 密集獎勵權重相加
    s0_complete_bonus      = RewTerm(func=mdp.s0_complete_bonus, weight=S0_COMPLETE_BONUS_WEIGHT)

    # ── Stage 1：移動到瓶口 + 花朵 +X 朝上 ──────────────────────────────
    s1_approach_bottle = RewTerm(func=mdp.s1_approach_bottle, weight=_S1_DENSE_WEIGHTS[0])
    s1_align_flower_up = RewTerm(func=mdp.s1_align_flower_up, weight=_S1_DENSE_WEIGHTS[1])
    # dead_penalty = -λ × cum_end(300+200=500)，λ=1
    s1_dead_penalty = RewTerm(func=mdp.s1_dead_penalty, weight=-500.0)
    # complete_bonus 權重 = S1 密集獎勵權重相加
    s1_complete_bonus = RewTerm(func=mdp.s1_complete_bonus, weight=S1_COMPLETE_BONUS_WEIGHT)

    # ── Stage 2：插入瓶子 ────────────────────────────────────────────────
    s2_approach_inside = RewTerm(func=mdp.s2_approach_inside, weight=_S2_DENSE_WEIGHTS[0])

    s2_release = RewTerm(
        func=mdp.s2_release,
        weight=MISSING,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )

    # dead_penalty = -λ × cum_end(200)，λ=1
    s2_dead_penalty   = RewTerm(func=mdp.s2_dead_penalty, weight=-200.0)
    # complete_bonus 權重 = S2 密集獎勵權重相加
    s2_complete_bonus = RewTerm(func=mdp.s2_complete_bonus, weight=S2_COMPLETE_BONUS_WEIGHT)

    # ── All stages ──────────────────────────────────────────────────────
    # 全任務完成（Stage 2 完成當步）獨立速度獎勵：weight × (所有階段總步數1000 - 目前 step)
    # weight = S0 + S1 + S2 所有密集獎勵權重相加
    all_complete_bonus = RewTerm(func=mdp.all_complete_bonus, weight=ALL_COMPLETE_BONUS_WEIGHT)
    # 動作平滑懲罰
    all_action_rate_l2 = RewTerm(func=mdp.all_action_rate_l2, weight=-0.001)
    all_joint_vel_l2   = RewTerm(func=mdp.all_joint_vel_l2, weight=-0.001)




@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # epoch 350 後（350 × 192 = 67200 steps）將 action_rate 和 joint_vel 懲罰加重 10 倍
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "all_action_rate_l2", "weight": -0.01, "num_steps": 67200},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "all_joint_vel_l2", "weight": -0.01, "num_steps": 67200},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)   # 1032 steps 安全網

    # Stage 0 預算(300步)耗盡仍未完成 → 死亡，並給予 -500 懲罰
    phase0_dead = DoneTerm(func=mdp.phase0_dead_termination, time_out=False)

    # Stage 1 預算(120步)耗盡仍未完成 → 死亡，並給予 -300 懲罰
    s1_dead = DoneTerm(func=mdp.s1_dead_termination, time_out=False)

    # Stage 2 預算(180步)耗盡仍未完成 → 死亡，並給予 -200 懲罰
    s2_dead = DoneTerm(func=mdp.s2_dead_termination, time_out=False)

    # Stage 2 完成 → 任務成功，結束回合
    task_success = DoneTerm(func=mdp.task_success_termination, time_out=False)


##
# Environment configuration
##


@configclass
class FlowerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Flower environment."""
    """櫃子環境的總配置類別。"""

    # 場景設定：同時啟動 4096 個平行環境，彼此間隔 2 公尺
    # Scene settings
    scene: FlowerSceneCfg = FlowerSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        """初始化後的參數微調。"""
        # general settings
        self.decimation = 1                 # 控制與模擬的頻率比（1 代表每一幀都控制）
        self.episode_length_s = 17.2       # 1032 steps @ 60Hz（stage budget 總和 1000 步 + 32 步緩衝，避免被全域 time_out 搶先攔截 s2 判斷）
        # 設定視窗攝影機位置
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        # 物理模擬參數
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 81920
        self.sim.physx.gpu_collision_stack_size = 134217728  # 128MB
