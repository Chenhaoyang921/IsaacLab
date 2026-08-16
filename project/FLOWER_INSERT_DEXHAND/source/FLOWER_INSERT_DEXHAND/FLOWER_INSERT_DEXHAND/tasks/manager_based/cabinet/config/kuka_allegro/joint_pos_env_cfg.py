# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from FLOWER_INSERT_DEXHAND.tasks.manager_based.cabinet import mdp
from FLOWER_INSERT_DEXHAND.tasks.manager_based.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    FlowerEnvCfg,
)

# Allegro 手的全部 16 個指關節
HAND_JOINTS = ["(index|middle|ring|thumb)_joint_(0|1|2|3)"]



@configclass
class KukaAllegroCabinetEnvCfg(FlowerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # KUKA iiwa7 + Allegro Hand 取代 Franka Panda
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        # 手臂：RL 正常控制
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["iiwa7_joint_.*"],
            scale=1.0,
            use_default_offset=True,
        )
        # 手掌：沿用二元開合語意，張開 / 握拳兩種姿態，RL 全程自由控制
        self.actions.gripper_action = mdp.CooldownBinaryGripperActionCfg(
            asset_name="robot",
            joint_names=HAND_JOINTS,
            open_command_expr={
                "(index|middle|ring)_joint_0": 0.0,
                "(index|middle|ring)_joint_(1|2|3)": 0.0,
                "thumb_joint_0": 1.5,
                "thumb_joint_(1|2|3)": 0.0,
            },
            close_command_expr={
                "(index|middle|ring)_joint_0": 0.0,
                "(index|middle|ring)_joint_(1|2|3)": 1.2,
                "thumb_joint_0": 1.5,
                "thumb_joint_(1|2|3)": 1.0,
            },
        )

        # 兩指指尖的接觸感測，取代原本的 panda_.*finger
        # 只取 index / thumb 兩根，is_catch() 預期 net_forces_w 的形狀是 (num_envs, 2, 3)
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/(index|thumb)_biotac_tip",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/iiwa7_link_0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/palm_link",
                    name="ee_tcp",
                    # 從掌心往手指方向偏移，對應 Franka 的 panda_hand -> TCP 偏移
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.10),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/index_biotac_tip",
                    name="tool_leftfinger",
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/thumb_biotac_tip",
                    name="tool_rightfinger",
                ),
            ],
        )

        # override rewards：把原本指向 panda_finger 的 asset_cfg 改成 Allegro 指關節
        self.rewards.s2_release.weight = 0.0
        self.rewards.s2_release.params["asset_cfg"].joint_names = HAND_JOINTS


@configclass
class KukaAllegroCabinetEnvCfg_PLAY(KukaAllegroCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
