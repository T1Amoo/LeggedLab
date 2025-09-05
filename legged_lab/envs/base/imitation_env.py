from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.imitation_env_config import ImitationEnvCfg
from legged_lab.envs.base.base_env_config import BaseEnvCfg

import os
import torch
import numpy as np
import csv

class ImitationEnv(BaseEnv):
    """
    Base class for imitation learning environments.
    This class extends the BaseEnv class to include specific configurations and functionalities
    for environments designed for imitation learning tasks.
    """
    def __init__(self, cfg: BaseEnvCfg, headless):
        super().__init__(cfg, headless)
        self.traj = self.load_csv_traj("/home/hiyio/LAFAN1_Retargeting_Dataset/g1/dance1_subject1.csv")
        self.traj_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def pre_physics_step(self, actions):
        traj_idx = torch.clamp(self.traj_indices, max=self.traj.shape[0] - 1)
        self.reward_manager.set_param("target_joint_traj", self.traj[traj_idx])
        print("refernce = ", self.traj[traj_idx])
        super().pre_physics_step(actions)

    def post_physics_step(self):
        self.traj_indices += 1
        self.traj_indices[self.reset_buf == 1] = 0
        super().post_physics_step()
    
    def load_csv_traj(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        # 使用 numpy 读入
        traj_array = np.loadtxt(path, delimiter=",")
        # 转换成 torch tensor
        traj_tensor = torch.tensor(traj_array, dtype=torch.float32, device=self.device)
        if traj_tensor.ndim != 2:
            raise ValueError(f"Trajectory tensor must be 2D, got shape {traj_tensor.shape}")
        return traj_tensor