# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    nf_hist = cs.data.net_forces_w_history  # [T, N, feet, 3]
    # 用“当前帧”的接触力度更靠谱（或用 current_contact_time>0 判接触）
    nf_now = cs.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)  # [N, feet]
    is_contact = (nf_now > threshold)
    return (torch.sum(is_contact, dim=-1) < 1).float()  # <1 表示“不是双脚都在接触”


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


# def feet_slide(
#     env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
#     asset: Articulation = env.scene[asset_cfg.name]
#     body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
#     reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
#     return reward
def feet_slide(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (cs.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0)  # [N, feet]

    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    cmd = env.command_generator.command[:, :2]
    heading = cmd / (cmd.norm(dim=1, keepdim=True) + 1e-6)
    v_parallel = (v * heading[:, None, :]).sum(dim=-1, keepdim=True) * heading[:, None, :]
    v_lateral = v - v_parallel
    return torch.sum(v_lateral.norm(dim=-1) * contacts, dim=1)



def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)

def joint_imitation_l1(env: BaseEnv, target_joint_traj: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    计算当前关节电机位置与目标轨迹的 L1 模仿损失奖励。

    参数:
        env: 当前环境
        target_joint_traj: [num_envs, num_joints] 的目标关节位置轨迹
        asset_cfg: 指定机器人及其关节信息的配置

    返回:
        [num_envs] 的奖励张量（越接近目标轨迹奖励越高）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    diff = torch.abs(current_pos - target_joint_traj)
    return torch.sum(diff, dim=1)

def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)

#TODO
# 在每帧更新最低身体部位的最大高度，但不返回 reward
# def update_min_body_height(env):
#     body_names = env.robot.data.body_names
#     link_zs = [env.robot.data.body_pos_w[env.robot.body_name_to_index(name)][2] for name in body_names]
#     current_min_z = min(link_zs)

#     if not hasattr(env, "_max_min_body_height"):
#         env._max_min_body_height = current_min_z
#     else:
#         env._max_min_body_height = max(env._max_min_body_height, current_min_z)

#     return env._max_min_body_height # 每帧 reward 不给值，只缓存数据

def update_min_body_height(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # world-frame 下各 link 的 z 轴高度： (num_envs, num_bodies)
    zs = asset.data.body_pos_w[:, :, 2]
    # 每个 env 的当前最低身高： (num_envs,)
    current_min_z, _ = torch.min(zs, dim=1)
    # 缓存每个 env 跳跃过程中的“最大最低身高”
    if not hasattr(env, "_max_min_body_height"):
        env._max_min_body_height = current_min_z.clone()
    else:
        env._max_min_body_height = torch.maximum(
            env._max_min_body_height,
            current_min_z,
        )
    # 返回 (num_envs,) 的张量；weight=0 时只做缓存
    return env._max_min_body_height

# 在终止时读取缓存，作为 episode 末尾一次性奖励
# def final_min_body_height_reward(env):
#     if getattr(env, "episode_length_buf", None) is not None:
#         done_envs = env.episode_length_buf >= env.max_episode_length  # 或使用 env.reset_buf
#         reward = torch.zeros(env.num_envs, device=env.device)
#         for i in range(env.num_envs):
#             if done_envs[i] and hasattr(env, "_max_min_body_height"):
#                 reward[i] = env._max_min_body_height  # 可换成 reward[i] = ...
#         return reward
#     else:
#         return 0.0

def final_min_body_height_reward(env: BaseEnv) -> torch.Tensor:
    """
    在每个 step 调用，只有当子环境被标记为 reset（reset_buf=True）时，
    才把缓存的 _max_min_body_height 发放为 reward，并立即清零对应缓存。
    """
    # 初始化全零奖励
    reward = torch.zeros(env.num_envs, device=env.device)
    # 确保缓存已存在
    if hasattr(env, "_max_min_body_height"):
        # 找出本帧需要重置的 env 索引
        reset_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            # 在重置那一刻，把缓存值发放给 reward
            reward[reset_ids] = env._max_min_body_height[reset_ids]
            # 并立即清零这些子环境的缓存，为下个 episode 重新累积
            env._max_min_body_height[reset_ids] = 0.0
    return reward  # 返回 (num_envs,) 张量

def feet_air_time_running(env: BaseEnv, sensor_cfg: SceneEntityCfg, 
                          stance_bonus_cap: float = 0.22,   # 单脚支撑奖励上限(秒)
                          flight_min: float = 0.03,         # 飞行最短阈(秒) ~ 1~3个控制周期
                          flight_max: float = 0.12,         # 飞行最长阈(秒) 按 5 m/s 合理的空中期
                          flight_scale: float = 1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]          # 每只脚的空中累计
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # 每只脚的接触累计
    in_contact = contact_time > 0.0
    # ① 单脚支撑奖励（与原逻辑一致，但封顶更紧）
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    stance_rew = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    stance_rew = torch.clamp(stance_rew, max=stance_bonus_cap)

    # ② 双脚同时离地的“短暂飞行”奖励（只在合理窗口内给奖励）
    both_air = torch.sum(in_contact.int(), dim=1) == 0
    # 取两脚的最小空中时间，代表“整机离地了多久”
    min_air = torch.min(air_time, dim=1)[0]
    flight_dur = torch.where(both_air, min_air, torch.zeros_like(min_air))
    flight_rew = torch.clamp(flight_dur - flight_min, min=0.0)
    # 超过 flight_max 的部分不再增加奖励
    flight_rew = torch.clamp(flight_rew, max=(flight_max - flight_min)) * flight_scale

    # 不在高速度就不给飞行奖励（避免低速乱跳）
    speed = torch.norm(env.command_generator.command[:, :2], dim=1)
    flight_gate = (speed > 1.5).float()  # >2 m/s 才算
    flight_rew = flight_rew * flight_gate

    # 合并
    return stance_rew + flight_rew

def lin_vel_z_l2_phase_gated(env, sensor_cfg=SceneEntityCfg("contact_sensor"),
                             asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    vz2 = asset.data.root_lin_vel_b[:, 2]**2
    cs = env.scene.sensors[sensor_cfg.name]
    in_contact_any = (cs.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).any(dim=1).float()
    return vz2 * in_contact_any

def body_force_speed_adaptive(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    base: float = 900.0,       # 静态阈值基线
    per_ms: float = 120.0,     # 每 1 m/s 额外增加的阈值
    max_reward: float = 400.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 这里沿用你原来的“取竖直分量”的写法
    fz = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs()  # [N, feet]

    # 速度门控：速度越快，阈值越高
    speed = torch.norm(env.command_generator.command[:, :2], dim=1, keepdim=True)  # [N,1]
    thr = base + per_ms * speed  # [N,1]

    # 仅惩罚超出阈值的部分，并做上限裁剪
    excess = (fz - thr).clamp(min=0.0)
    reward = excess.sum(dim=1)                      # 合并双脚
    return reward.clamp(min=0.0, max=max_reward)