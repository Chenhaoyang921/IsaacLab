# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from FLOWER_INSERT.tasks.manager_based.cabinet import mdp
from FLOWER_INSERT.tasks.manager_based.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    FlowerEnvCfg,
)

##
# Pre-defined configs
##
from FLOWER_INSERT.tasks.manager_based.cabinet.assets.franka_local import FRANKA_PANDA_LOCAL_CFG  # isort: skip


@configclass
class FrankaCabinetEnvCfg(FlowerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot（使用本地 USD，不依賴 Nucleus 伺服器）
        self.scene.robot = FRANKA_PANDA_LOCAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 開啟右手指的接觸感測，偵測對花朵的夾持力
        self.scene.robot.spawn.activate_contact_sensors = True

        # 手臂：RL 正常控制
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        # 夾爪：RL 控制開合（epoch<300 無冷卻，epoch≥300 切換後鎖定 0.1 秒）
        self.actions.gripper_action = mdp.CooldownBinaryGripperActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": -0.0005},
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        # override rewards
        self.rewards.s2_release.weight = 0.0
        self.rewards.s2_release.params["asset_cfg"].joint_names = ["panda_finger_.*"]


@configclass
class FrankaCabinetEnvCfg_PLAY(FrankaCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
