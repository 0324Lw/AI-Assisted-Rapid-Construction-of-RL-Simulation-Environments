import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import math


class Config:
    """保存所有可自定义的参数配置"""

    def __init__(self):
        # 物理环境参数
        self.map_size = 10.0  # 地图边长 (10m x 10m)
        self.num_obstacles = 8  # 障碍物数量
        self.obs_radius_range = (0.3, 0.8)  # 障碍物半径范围 (正方形已等效为外接圆)
        self.goal_tolerance = 0.4  # 判定到达终点的距离阈值

        # 智能体运动学参数
        self.dt = 0.1  # 控制周期 (秒)
        self.v_max = 1.0  # 最大线速度 (m/s)
        self.omega_max = np.pi / 2  # 最大角速度 (rad/s)
        self.max_steps = 400  # 每回合最大步数

        # 激光雷达 (LiDAR) 参数
        self.lidar_rays = 24  # 激光射线数量
        self.lidar_max_range = 3.5  # 最大探测距离 (m)

        # 奖励函数系数微调版
        self.w_step = -0.02  # 步数惩罚 (保持不变)
        self.w_approach = 5.0  # [修改] 从 10.0 降至 5.0，单步最大奖励约 0.5
        self.w_orientation = 0.2  # [修改] 从 0.1 升至 0.2，增强车头对准的引导
        self.w_smooth = -0.015  # [修改] 从 -0.05 降至 -0.015，大幅降低改变动作的顾虑
        self.r_collision = -200.0  # 碰撞惩罚 (保持不变)
        self.r_goal = 200.0  # 终点奖励 (保持不变)


class Env(gym.Env):
    """强化学习环境主体"""

    def __init__(self, config=Config()):
        super(Env, self).__init__()
        self.cfg = config

        # 定义动作空间: [线速度, 角速度]，范围归一化至 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 定义状态空间: LiDAR (N维) + GoalDist (1) + GoalAngle (1) + v (1) + omega (1)
        state_dim = self.cfg.lidar_rays + 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)

        # 初始化环境内部变量
        self.obstacles = []  # 存储 (x, y, r)
        self.agent_pos = np.zeros(2)
        self.agent_theta = 0.0
        self.goal_pos = np.zeros(2)
        self.current_v = 0.0
        self.current_omega = 0.0
        self.steps = 0
        self.dist_to_goal_prev = 0.0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.steps = 0
        self.current_v = 0.0
        self.current_omega = 0.0

        # 1. 随机生成障碍物 (保证不越界)
        self.obstacles = []
        for _ in range(self.cfg.num_obstacles):
            r = np.random.uniform(self.cfg.obs_radius_range[0], self.cfg.obs_radius_range[1])
            x = np.random.uniform(r, self.cfg.map_size - r)
            y = np.random.uniform(r, self.cfg.map_size - r)
            self.obstacles.append((x, y, r))

        # 2. 生成起点和终点 (确保不在障碍物内，且有一定初始距离)
        self.agent_pos = self._generate_valid_pos()
        self.agent_theta = np.random.uniform(-np.pi, np.pi)

        while True:
            self.goal_pos = self._generate_valid_pos()
            if np.linalg.norm(self.agent_pos - self.goal_pos) > 4.0:  # 起点终点至少相距 4m
                break

        self.dist_to_goal_prev = np.linalg.norm(self.agent_pos - self.goal_pos)
        return self._get_state(), {}

    def step(self, action):
        self.steps += 1

        # 1. 解析动作并反归一化物理量 (v不允许后退)
        a_v, a_omega = np.clip(action, -1.0, 1.0)
        v = ((a_v + 1.0) / 2.0) * self.cfg.v_max
        omega = a_omega * self.cfg.omega_max

        # 计算动作变化惩罚
        delta_v = v - self.current_v
        delta_omega = omega - self.current_omega
        r_smooth = self.cfg.w_smooth * (delta_v ** 2 + delta_omega ** 2)

        self.current_v = v
        self.current_omega = omega

        # 2. 运动学更新
        self.agent_pos[0] += v * np.cos(self.agent_theta) * self.cfg.dt
        self.agent_pos[1] += v * np.sin(self.agent_theta) * self.cfg.dt
        self.agent_theta += omega * self.cfg.dt

        # 将角度限制在 [-pi, pi]
        self.agent_theta = (self.agent_theta + np.pi) % (2 * np.pi) - np.pi

        # 3. 计算状态与奖励信息
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)

        # 预计算 info 字典记录
        info = {
            'r_step': self.cfg.w_step,
            'r_approach': 0.0,
            'r_orientation': 0.0,
            'r_smooth': float(r_smooth),
            'r_collision': 0.0,
            'r_goal': 0.0
        }

        reward = self.cfg.w_step + r_smooth
        done = False
        truncated = False

        # 检查碰撞
        if self._check_collision():
            reward += self.cfg.r_collision
            info['r_collision'] = self.cfg.r_collision
            done = True
        # 检查到达终点
        elif dist_to_goal < self.cfg.goal_tolerance:
            reward += self.cfg.r_goal
            info['r_goal'] = self.cfg.r_goal
            done = True
        # 正常行驶中的密集奖励
        else:
            # 靠近奖励
            r_app = self.cfg.w_approach * (self.dist_to_goal_prev - dist_to_goal)
            reward += r_app
            info['r_approach'] = float(r_app)

            # 方向奖励
            vec_to_goal = self.goal_pos - self.agent_pos
            target_theta = np.arctan2(vec_to_goal[1], vec_to_goal[0])
            theta_error = target_theta - self.agent_theta
            theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
            r_ori = self.cfg.w_orientation * (1.0 - abs(theta_error) / np.pi)
            reward += r_ori
            info['r_orientation'] = float(r_ori)

        self.dist_to_goal_prev = dist_to_goal

        if self.steps >= self.cfg.max_steps and not done:
            truncated = True

        # 限制单步 reward 范围
        reward = np.clip(reward, -2.0, 2.0)
        # 碰撞和到达终点覆盖单步限制以给予强信号
        if done:
            reward = info['r_collision'] if info['r_collision'] != 0 else info['r_goal']

        return self._get_state(), reward, done, truncated, info

    def _get_state(self):
        """获取归一化后的状态向量"""
        # 1. LiDAR 测距 [0, 1]
        lidar_data = self._compute_lidar() / self.cfg.lidar_max_range

        # 2. 相对目标极坐标
        vec_to_goal = self.goal_pos - self.agent_pos
        dist = np.linalg.norm(vec_to_goal)
        norm_dist = np.clip(dist / (self.cfg.map_size * 1.414), 0.0, 1.0)  # 最大对角线归一化

        target_theta = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        theta_err = (target_theta - self.agent_theta + np.pi) % (2 * np.pi) - np.pi
        norm_theta_err = theta_err / np.pi  # [-1, 1]

        # 3. 当前速度归一化
        norm_v = self.current_v / self.cfg.v_max  # [0, 1]
        norm_w = self.current_omega / self.cfg.omega_max  # [-1, 1]

        state = np.concatenate([lidar_data, [norm_dist, norm_theta_err, norm_v, norm_w]])
        return state.astype(np.float32)

    def _compute_lidar(self):
        """模拟 2D 激光雷达扫描 (采用线与圆相交的解析解加速计算)"""
        angles = np.linspace(0, 2 * np.pi, self.cfg.lidar_rays, endpoint=False) + self.agent_theta
        distances = np.full(self.cfg.lidar_rays, self.cfg.lidar_max_range)

        for i, angle in enumerate(angles):
            dx = np.cos(angle)
            dy = np.sin(angle)

            # 检查与地图边界的交点
            t_bounds = []
            if dx > 1e-6:
                t_bounds.append((self.cfg.map_size - self.agent_pos[0]) / dx)
            elif dx < -1e-6:
                t_bounds.append(-self.agent_pos[0] / dx)
            if dy > 1e-6:
                t_bounds.append((self.cfg.map_size - self.agent_pos[1]) / dy)
            elif dy < -1e-6:
                t_bounds.append(-self.agent_pos[1] / dy)

            t_wall = min([t for t in t_bounds if t > 0] + [self.cfg.lidar_max_range])
            distances[i] = t_wall

            # 检查与障碍物的交点 (解一元二次方程)
            for ox, oy, r in self.obstacles:
                vx = self.agent_pos[0] - ox
                vy = self.agent_pos[1] - oy

                b = 2.0 * (vx * dx + vy * dy)
                c = (vx ** 2 + vy ** 2) - r ** 2
                discriminant = b ** 2 - 4 * c

                if discriminant >= 0:
                    t1 = (-b - np.sqrt(discriminant)) / 2.0
                    t2 = (-b + np.sqrt(discriminant)) / 2.0

                    if t1 > 0 and t1 < distances[i]:
                        distances[i] = t1
                    elif t2 > 0 and t2 < distances[i] and t1 <= 0:
                        distances[i] = t2

        return distances

    def _check_collision(self):
        """检测是否发生碰撞"""
        x, y = self.agent_pos
        # 越界检测
        if x < 0 or x > self.cfg.map_size or y < 0 or y > self.cfg.map_size:
            return True
        # 障碍物碰撞检测
        for ox, oy, r in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= r ** 2:
                return True
        return False

    def _generate_valid_pos(self):
        """生成无碰撞的随机坐标"""
        while True:
            pos = np.random.uniform(0, self.cfg.map_size, size=2)
            collision = False
            for ox, oy, r in self.obstacles:
                if np.linalg.norm(pos - np.array([ox, oy])) <= (r + 0.2):  # 预留0.2m安全余量
                    collision = True
                    break
            if not collision:
                return pos


class Plot:
    """通用数据与轨迹可视化类"""

    @staticmethod
    def plot_learning_curve(rewards, window=50):
        """绘制训练曲线及移动平均线"""
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')

        # 计算滑动平均
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
            plt.plot(np.arange(window - 1, len(rewards)), moving_avg, color='red', label=f'{window}-Episode Moving Avg')

        plt.title('Training Learning Curve')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_trajectory(env, trajectory):
        """绘制验证或测试时的运动轨迹"""
        plt.figure(figsize=(8, 8))

        # 绘制终点
        plt.scatter(env.goal_pos[0], env.goal_pos[1], c='red', marker='*', s=200, label='Goal')

        # 绘制障碍物
        ax = plt.gca()
        for ox, oy, r in env.obstacles:
            circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5)
            ax.add_patch(circle)

        # 绘制轨迹
        traj = np.array(trajectory)
        if len(traj) > 0:
            plt.plot(traj[:, 0], traj[:, 1], c='blue', linestyle='--', label='Trajectory')
            plt.scatter(traj[0, 0], traj[0, 1], c='green', s=100, label='Start')  # 起点

        plt.xlim(0, env.cfg.map_size)
        plt.ylim(0, env.cfg.map_size)
        plt.title('Agent Navigation Trajectory')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.grid(True)
        plt.show()