from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BlackCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        # 1. 初始姿态
        pos = [0.0, 0.0, 0.50] 
        default_joint_angles = {
            'FL_hip_joint': 0.0,   'FL_thigh_joint': 0.82,   'FL_calf_joint': -1.5,
            'FR_hip_joint': -0.0,  'FR_thigh_joint': -0.82,  'FR_calf_joint': 1.5,
            'RL_hip_joint': 0.0,   'RL_thigh_joint': 0.82,   'RL_calf_joint': -1.5,
            'RR_hip_joint': -0.0,  'RR_thigh_joint': -0.82,  'RR_calf_joint': 1.5
        }

    class control(LeggedRobotCfg.control):
        # 2. PD 参数
        # 刚度 (P Gain)
        stiffness = {
            'FL_hip_joint': 40.0, 'RL_hip_joint': 40.0, 'FR_hip_joint': 40.0, 'RR_hip_joint': 40.0,
            'FL_thigh_joint': 40.0, 'RL_thigh_joint': 40.0, 'FR_thigh_joint': 40.0, 'RR_thigh_joint': 40.0,
            'FL_calf_joint': 40.0, 'RL_calf_joint': 40.0, 'FR_calf_joint': 40.0, 'RR_calf_joint': 40.0
        }
        # 阻尼 (D Gain)
        damping = {
            'FL_hip_joint': 1.0, 'RL_hip_joint': 1.0, 'FR_hip_joint': 1.0, 'RR_hip_joint': 1.0,
            'FL_thigh_joint': 1.0, 'RL_thigh_joint': 1.0, 'FR_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
            'FL_calf_joint': 1.0, 'RL_calf_joint': 1.0, 'FR_calf_joint': 1.0, 'RR_calf_joint': 1.0
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        # 3. 指定 URDF 路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/black/urdf/black_description.urdf'
        name = "black"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base", "thigh"]
        self_collisions = 1 # 1=disable

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
  
    class env(LeggedRobotCfg.env):
        num_envs = 3400
        num_observations = 45 * 6 # frame_stack=6 (默认)
        
        # 保持 HIMLoco 的 Estimator 需要的特权观测设置
        num_privileged_obs = 45 + 3 + 3 + 187

    class rewards(LeggedRobotCfg.rewards):
        cycle_time = 0.6
        clearance_height_target = 0.08
        soft_dof_pos_limit = 0.9
        base_height_target = 0.48
        only_positive_rewards = False
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -2.0
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.03
            action_rate = -0.05
            smoothness = -0.01
            feet_air_time = 0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.8
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            trot = 0.3
            hip_pos = -0.5
            all_joint_pos = -0.2
            foot_slip = -0.1
            #lateral_vel_penalty = -0.1

class BlackCfgPPO(LeggedRobotCfgPPO):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_black_dog'
        # 指定算法
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        max_iterations = 300