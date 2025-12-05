from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class BlackEnv(LeggedRobot):
    """
    自定义环境类 BlackEnv，适配 HIMLoco框架
    """

    def _init_buffers(self):
        """ 初始化 Buffer，额外获取所有刚体状态用于自定义奖励 """
        super()._init_buffers()

        # 获取所有刚体的状态(可用于计算脚部位置、速度等)
        # 形状：(num_envs, num_bodies, 13)
        # 13维包括：pos(3), quat(4), lin_vel(3), ang_vel(3)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)

    def post_physics_step(self):
        """ 物理步后刷新状态 """
        env_ids, termination_privileged_obs = super().post_physics_step()
        # 手动刷新刚体状态
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        return env_ids, termination_privileged_obs

    def _get_phase(self):
        """ 
        内部辅助函数，计算相位
        仅用于计算奖励函数，不作为观测输入给网络
        """

        cycle_time = self.cfg.rewards.cycle_time
        phase = (self.episode_length_buf * self.dt) % cycle_time / cycle_time
        return phase
    
    def _get_gait_phase(self):
        """
        根据相位生成理想的触地掩码 (Stance Mask)
        1 表示支撑相 (应触地)，0 表示摆动相 (应抬脚)
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        
        # 添加双支撑相 (Double Support Phase)
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        
        # 左腿支撑 (Left Stance) -> 对应 sin >= 0
        stance_mask[:, 0] = sin_pos >= 0
        # 右腿支撑 (Right Stance) -> 对应 sin < 0
        stance_mask[:, 1] = sin_pos < 0
        
        # 双支撑相：当 sin 值接近 0 时，两腿都应该着地
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    
    # ----------------------------------------------------------------------
    # 自定义奖励函数区域
    # ----------------------------------------------------------------------

    def _reward_trot(self):
        """
        [Trot 步态引导奖励]
        鼓励对角线脚同时接触地面，且符合目标相位
        """
        # 获取脚底 Z 轴接触力
        contact_force_z = self.contact_forces[:, self.feet_indices, 2]
        # 使用 sigmoid 将力转换为触地概率 (0~1)
        contact_prob = torch.sigmoid((contact_force_z - 5.0) * 0.5)
        
        fl, fr, rl, rr = contact_prob[:, 0], contact_prob[:, 1], contact_prob[:, 2], contact_prob[:, 3]
        
        # 1. 对角线同步奖励：FL 和 RR 应该状态一致，FR 和 RL 应该状态一致
        diag1_sync = 1.0 - torch.abs(fl - rr)
        diag2_sync = 1.0 - torch.abs(fr - rl)
        
        # 2. 计算每组对角线的平均触地情况
        s1 = 0.5 * (fl + rr) # 1号对角线 (FL+RR)
        s2 = 0.5 * (fr + rl) # 2号对角线 (FR+RL)
        
        # 3. 互斥奖励：确保两组对角线不同时触地，也不同时抬起 (s1 + s2 应该接近 1)
        phase_score = 1.0 - torch.abs((s1 + s2) - 1.0)
        
        # 4. 与目标相位匹配
        stance_mask = self._get_gait_phase().float()
        target_s1, target_s2 = stance_mask[:, 0], stance_mask[:, 1]
        match_s1 = 1.0 - torch.abs(s1 - target_s1)
        match_s2 = 1.0 - torch.abs(s2 - target_s2)
        
        # 组合各项分数
        alpha, beta = 0.24, 0.5
        diag1_score = alpha * diag1_sync + (1 - alpha) * match_s1
        diag2_score = alpha * diag2_sync + (1 - alpha) * match_s2
        
        base_rew = 0.5 * (diag1_score + diag2_score)
        rew = base_rew * (beta * phase_score)
        
        # 只有在有速度指令时才给予奖励 (静止时不需要踏步)
        rew = rew * (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        
        # 仅在有移动指令时生效
        return rew
    
    def _reward_foot_slip(self):
        """
        [脚底打滑惩罚]
        触地时如果脚有水平速度则惩罚
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm) * contact
        return torch.sum(rew, dim=1)
    
    def _reward_hip_pos(self):
        """ 
        [髋关节限位惩罚]
        惩罚髋关节 (Hip/Abduction) 偏离默认角度的程度。
        防止机器人两腿张得太开 (劈叉) 或向内收得太多。
        """

        hip_indices = [0, 3, 6, 9]

        return torch.sum(torch.abs(self.dof_pos[:, hip_indices] - self.default_dof_pos[:, hip_indices]), dim=1)

    def _reward_all_joint_pos(self):
        """
        [所有关节限位惩罚]
        惩罚所有关节偏离默认角度的程度
        防止动作变形
        """

        return torch.sum(torch.abs(self.dof_pos[:,:] - self.dof_pos[:,:]), dim=1)
    
    def _reward_lateral_vel_penalty(self):
        """
        [横向速度惩罚]
        指令为前进时惩罚横向速度
        """
        v_y = self.base_lin_vel[:, 1]

        # 获取横向移动指令
        cmd_y = self.commands[:, 1]

        # 判断是否应该直行
        is_straight_command = torch.abs(cmd_y) < 0.1

        # 计算惩罚
        penalty = torch.square(v_y) * is_straight_command.float()

        return penalty
