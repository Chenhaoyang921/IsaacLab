# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""自訂動作類別。

CooldownBinaryGripperAction — 夾爪：無階段版本，RL 自由控制開閉（無冷卻、無切換限額）
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass


# ──────────────────────────────────────────────────────────────────────────────
# 夾爪動作：自由控制（無階段版本）
# ──────────────────────────────────────────────────────────────────────────────

class CooldownBinaryGripperAction(BinaryJointPositionAction):
    """二元夾爪動作，無階段版本：RL 完全自由控制，不設冷卻時間與切換次數上限。"""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)


@configclass
class CooldownBinaryGripperActionCfg(BinaryJointPositionActionCfg):
    """CooldownBinaryGripperAction 的配置類別。"""
    class_type: type = CooldownBinaryGripperAction
