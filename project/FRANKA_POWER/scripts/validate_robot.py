# Copyright (c) 2026, FRANKA_POWER Project.
# SPDX-License-Identifier: BSD-3-Clause

"""驗證 FRANKA_POWER_ROBOT_CFG 這組機械手臂設定是否正確。

開 GUI，只 spawn 機器人本身（不含 IK、不含 RL 任務），讓你眼睛直接確認：
  1. IsaacLab 自動印出的 "Simulation Joint Information" 表格
     -> 核對 effort limit / stiffness / damping / position & velocity limits
        是不是真的照你在 franka_power_env_cfg.py 寫的生效
  2. 掛上 PAYLOAD_MASS 前後的質量/慣量變化（印在終端機）
  3. 機械手在「只用重力補償」下是否能撐住姿勢不塌陷
     （懸空、輕輕給隨機小力矩，看關節有沒有跑到極限或抖動炸開）

用法（先啟動 env_isaaclab venv，見 run_validate.ps1）:
    .\isaaclab.bat -p project\FRANKA_POWER\scripts\validate_robot.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="驗證 FRANKA_POWER 機械手臂設定。")
parser.add_argument("--payload_mass", type=float, default=1.0, help="末端模擬負載重量 (kg)，0 表示不掛負載。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "FRANKA_POWER"))
from FRANKA_POWER.tasks.direct.franka_power.franka_power_env_cfg import FRANKA_POWER_ROBOT_CFG  # noqa: E402


def design_scene() -> Articulation:
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = FRANKA_POWER_ROBOT_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)
    return robot


def check_payload(robot: Articulation, payload_mass: float):
    """驗證負載質量/慣量是否正確套用到 panda_hand。"""
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    masses_before = robot.root_physx_view.get_masses().clone()
    inertias_before = robot.root_physx_view.get_inertias().clone()
    print(f"[VALIDATE] panda_hand 原始質量: {masses_before[0, hand_idx].item():.4f} kg")

    if payload_mass > 0.0:
        masses = masses_before.clone()
        inertias = inertias_before.clone()
        hand_mass = masses[:, hand_idx].clone()
        ratio = (hand_mass + payload_mass) / hand_mass
        masses[:, hand_idx] += payload_mass
        inertias[:, hand_idx, :] *= ratio.unsqueeze(-1)
        all_ids = torch.arange(robot.num_instances)
        robot.root_physx_view.set_masses(masses, all_ids)
        robot.root_physx_view.set_inertias(inertias, all_ids)

        masses_after = robot.root_physx_view.get_masses()
        print(f"[VALIDATE] 掛上 {payload_mass} kg 負載後 panda_hand 質量: {masses_after[0, hand_idx].item():.4f} kg")
        print(f"[VALIDATE]   慣量放大比例: {ratio[0].item():.3f}x")


def run_simulator(sim: sim_utils.SimulationContext, robot: Articulation):
    sim_dt = sim.get_physics_dt()
    arm_joint_ids, arm_joint_names = robot.find_joints("panda_joint[1-7]")
    print(f"[VALIDATE] 手臂關節 (依序): {arm_joint_names}")
    print(f"[VALIDATE] 手臂關節 effort_limit (Nm): {robot.data.joint_effort_limits[0, arm_joint_ids].tolist()}")
    print(f"[VALIDATE] 手臂關節 position 限制: {robot.data.joint_pos_limits[0, arm_joint_ids].tolist()}")

    count = 0
    while simulation_app.is_running():
        if count % 300 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print("[VALIDATE] 重置關節到預設姿勢（檢查機械手在此姿勢下能否僅靠 PD 撐住，不塌陷）")
        else:
            # 什麼都不做，只用預設姿勢的 position target 撐住 -> 看 stiffness/damping 夠不夠對抗重力+負載
            robot.set_joint_position_target(
                robot.data.default_joint_pos[:, arm_joint_ids], joint_ids=arm_joint_ids
            )

        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.5], [0.3, 0.0, 0.5])

    robot = design_scene()
    sim.reset()

    check_payload(robot, args_cli.payload_mass)

    print("[VALIDATE] 設定完成，開始模擬。觀察機械手是否維持預設姿勢、無異常抖動或塌陷。")
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
