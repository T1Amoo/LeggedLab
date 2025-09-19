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

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import G1_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, PLANE_TERRAIN_CFG


@configclass
class G1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2_phase_gated, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=-2.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.15,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    # )
    feet_air_time = RewTerm(
    func=mdp.feet_air_time_running,
    weight=0.5,   # 初始 0.3~0.5；跑起来后可微调
    params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*")}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    # feet_force = RewTerm(
    #     func=mdp.body_force,
    #     weight=-3e-3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
    #         "threshold": 500,
    #         "max_reward": 400,
    #     },
    # )
    feet_force = RewTerm(
    func=mdp.body_force_speed_adaptive,
    weight=-3e-3,
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
        "base": 900.0, "per_ms": 120.0, "max_reward": 400.0
    },
)
    
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_wrist.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])},
    )

@configclass
class G1ImitationRewardCfg(RewardCfg):
    joint_imitation_hip = RewTerm(
        func=mdp.joint_imitation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"]
            ),
            "target_joint_traj": None
        },
    )
    joint_imitation_arms = RewTerm(
        func=mdp.joint_imitation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_wrist.*"]
            ),
            "target_joint_traj": None
        },
    )
    joint_imitation_legs = RewTerm(
        func=mdp.joint_imitation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"]), "target_joint_traj": None},
    )   
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    
@configclass
class G1FlatEnvCfg(BaseEnvCfg):

    reward = G1RewardCfg()
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        # === 速度与采样课程 ===
        self.commands.ranges.lin_vel_x = (-0.4, 2.0)      # 跑步上限
        self.commands.ranges.lin_vel_y = (-0.7, 0.7)
        self.commands.ranges.ang_vel_z = (-1.57, 1.57)
        self.commands.ranges.heading = (-math.pi, math.pi)
        self.commands.resampling_time_range = (3.0, 5.0) # 更频繁换指令以学过渡
        self.commands.heading_control_stiffness = 0.8

        # === 速度跟踪宽容度（高速度更宽）===
        # 简单做法：直接放宽 std；若做自适应，可在 mdp 里按 |cmd| 动态放大
        self.reward.track_lin_vel_xy_exp.params["std"] = 1.5
        self.reward.track_ang_vel_z_exp.params["std"] = 1.5
        self.reward.track_lin_vel_xy_exp.weight = 3.0
        self.reward.track_ang_vel_z_exp.weight = 3.0

        # === 能量/平滑正则 ===
        self.reward.energy.weight = -5e-4                # 起步强度，稳态可逐步 -8e-4 ~ -1.2e-3
        self.reward.action_rate_l2.weight = -0.004       # 高速先弱一点，防止限制回摆
        self.reward.dof_acc_l2.weight = -2e-6            # 稍加强，抑制猛加速引发过流

        # === 姿态与垂直扰动 ===
        self.reward.lin_vel_z_l2.weight = -0.005          # 垂直速度偏小，跑步更灵活
        self.reward.ang_vel_xy_l2.weight = -0.15
        self.reward.body_orientation_l2.weight = -0.5
        self.reward.flat_orientation_l2.weight = -0.2

        # === 空中相 vs. 飞行：避免奖励“全离地” ===
        self.reward.feet_air_time.weight = 0.25
        # 空中相上限可根据速度稍降到 0.20–0.25，更贴近高速步频；如需：self.reward.feet_air_time.params["threshold"] = 0.22
        self.reward.fly.weight = 0.02                    # 不奖励全脚腾空；或设为小负值以抑制“蹦跳”
        self.reward.fly.params["threshold"] = 40.0
        # === 接触/碰撞 ===
        # 1 N 阈值太低，容易把轻微触碰算“undesired”
        self.reward.undesired_contacts.params["threshold"] = 200.0
        self.reward.feet_too_near.weight = -1.0
        self.reward.termination_penalty.weight = -100.0

        # === 观测长度 ===
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        

@configclass
class G1FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_flat"
    wandb_project: str = "g1_flat"


@configclass
class G1RoughEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25


@configclass
class G1RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_rough"
    wandb_project: str = "g1_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"

"""
实现模仿学习
"""
@configclass
class G1ImitationEnvCfg(BaseEnvCfg):

    reward = G1ImitationRewardCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1

@configclass
class G1ImitationAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_imitation"
    wandb_project: str = "g1_imitation"


@configclass
class G1JumpEnvCfg(BaseEnvCfg):
    # 指定使用的奖励配置，定义了多种 reward term（如速度跟踪、能耗、姿态等）
    
    reward = G1RewardCfg()
 

    # 每步更新高度缓存（不产生 reward）
    reward.update_min_body_height = RewTerm(
        func=mdp.update_min_body_height,
        weight=0.0,
    )

    # 仅在终止时给予一次性奖励（用 max_min_height）
    reward.final_min_body_height_reward = RewTerm(
        func=mdp.final_min_body_height_reward,
        weight=100.0,  # 奖励力度，可调
    )

    def __post_init__(self):
        super().__post_init__()
        self.reward.track_lin_vel_xy_exp.weight      = 0.0
        self.reward.track_ang_vel_z_exp.weight       = 0.0
        self.reward.lin_vel_z_l2.weight              = 0.0
        self.reward.ang_vel_xy_l2.weight             = 0.0
        self.reward.energy.weight                    = 0.0
        self.reward.dof_acc_l2.weight                = -2.5e-7
        self.reward.action_rate_l2.weight            = -0.05
        self.reward.undesired_contacts.weight        = -1.0
        self.reward.fly.weight                       = 1.0
        self.reward.body_orientation_l2.weight       = 0.0
        self.reward.flat_orientation_l2.weight       = 0.0
        self.reward.termination_penalty.weight       = -200.0
        self.reward.feet_air_time.weight             = 0.0
        self.reward.feet_slide.weight                = 0.0
        # self.reward.feet_force.weight                = 0.0
        self.reward.feet_too_near.weight             = 0.0
        self.reward.feet_stumble.weight              = 0.0
        self.reward.dof_pos_limits.weight            = 0.0
        self.reward.joint_deviation_hip.weight       = 0.0
        self.reward.joint_deviation_arms.weight      = 0.0
        self.reward.joint_deviation_legs.weight      = 0.0
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        # 设置高度扫描模块扫描的参考刚体为 torso（躯干），用于地形感知等功能
        self.scene.height_scanner.prim_body_name = "torso_link"

        # 设置机器人模型配置为 G1（定义了机器人 USD 路径、控制器、关节名等）
        self.scene.robot = G1_CFG

        # 设置地形类型为生成器模式（可动态创建不同地形），而不是静态平面或网格
        self.scene.terrain_type = "generator"

        # 使用碎石地形生成器配置，提供不平坦的地面用于训练步态鲁棒性
        self.scene.terrain_generator = PLANE_TERRAIN_CFG

        # # 设置终止接触体为 torso，一旦 torso 接触地面，则 episode 被判定为失败终止
        # self.robot.terminate_contacts_body_names = [".*torso.*"]

        # 终止条件：除了脚底（ankle_roll）外，其他 body 一旦接地就终止
        self.robot.terminate_contacts_body_names = ["^(?!.*ankle_roll).*"]
        
        # 设置足部 body 名称，用于检测脚部是否接地、滑动、起跳等状态
        self.robot.feet_body_names = [".*ankle_roll.*"]

        # 配置领域随机化中“质量扰动事件”的作用目标为 torso
        # 即训练过程中会对躯干质量进行随机扰动，以提升策略的鲁棒性
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]

        # 降低 domain random，方便学习跳跃
        self.domain_rand.enable = True
        
        self.commands.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.heading = (0.0, 0.0)
        
       
@configclass
class G1JumpAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_jump"
    wandb_project: str = "g1_jump"

    def __post_init__(self):
        super().__post_init__()

        # self.policy.class_name = "ActorCriticRecurrent"
        # self.policy.actor_hidden_dims = [256, 256, 128]
        # self.policy.critic_hidden_dims = [256, 256, 128]
        # self.policy.rnn_hidden_size = 256
        # self.policy.rnn_num_layers = 1
        # self.policy.rnn_type = "lstm"