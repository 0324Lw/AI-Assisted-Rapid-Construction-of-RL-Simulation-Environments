import numpy as np
import matplotlib.pyplot as plt
from env import Env, Config


def test_spaces_and_interaction(env):
    """测试 1: 状态空间、动作空间与环境交互测试"""
    print("=" * 50)
    print("1. 开始测试状态空间、动作空间与环境交互")

    state, info = env.reset()
    print(f"[*] 初始状态维度: {state.shape}, 数据类型: {state.dtype}")
    assert env.observation_space.contains(state), "初始状态不在观测空间范围内！"

    action = env.action_space.sample()
    print(f"[*] 采样随机动作: {action}, 维度: {action.shape}")
    assert env.action_space.contains(action), "采样动作不在动作空间范围内！"

    next_state, reward, done, truncated, info = env.step(action)
    print(f"[*] 执行单步后 - 奖励: {reward:.4f}, Done: {done}, Truncated: {truncated}")
    print(f"[*] 包含的 Info 键值: {list(info.keys())}")
    print("[+] 接口与空间测试通过！\n")


def test_generate_10_maps(env):
    """测试 2: 随机生成 10 张环境二维平面图"""
    print("=" * 50)
    print("2. 正在生成 10 张随机环境地图...")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("10 Randomly Generated Environment Maps", fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        env.reset()

        # 绘制终点 (红色五角星)
        ax.scatter(env.goal_pos[0], env.goal_pos[1], c='red', marker='*', s=150, zorder=3)
        # 绘制起点 (绿色圆点)
        ax.scatter(env.agent_pos[0], env.agent_pos[1], c='green', marker='o', s=80, zorder=3)

        # 绘制障碍物 (灰色圆形)
        for ox, oy, r in env.obstacles:
            circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5, zorder=2)
            ax.add_patch(circle)

        ax.set_xlim(0, env.cfg.map_size)
        ax.set_ylim(0, env.cfg.map_size)
        ax.set_title(f"Map {i + 1}")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    print("[+] 地图生成测试完成！请查看弹出的绘图窗口。\n")


def test_reward_components(env):
    """测试 3: 场景测试与奖励组件数据分析"""
    print("=" * 50)
    print("3. 开始场景测试与奖励组件数据分析 (共 3000 步)...")

    all_infos = {
        'r_step': [], 'r_approach': [], 'r_orientation': [],
        'r_smooth': [], 'r_collision': [], 'r_goal': []
    }

    # 场景 A: 随机探索 (1000步)
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        _, _, done, _, info = env.step(action)
        for k in all_infos.keys():
            all_infos[k].append(info[k])
        if done:
            env.reset()

    # 场景 B: 靠近障碍物 (1000步) - 强行将智能体放置在障碍物边缘并向其运动
    for _ in range(1000):
        env.reset()
        ox, oy, r = env.obstacles[np.random.randint(0, len(env.obstacles))]
        # 将智能体放在距离障碍物中心 r + 0.1 的位置
        angle_to_obs = np.random.uniform(-np.pi, np.pi)
        env.agent_pos = np.array([ox + (r + 0.1) * np.cos(angle_to_obs),
                                  oy + (r + 0.1) * np.sin(angle_to_obs)])
        env.agent_theta = angle_to_obs + np.pi  # 面向障碍物
        env.dist_to_goal_prev = np.linalg.norm(env.agent_pos - env.goal_pos)

        action = np.array([1.0, 0.0])  # 强制全速前进引发碰撞或靠近
        _, _, _, _, info = env.step(action)
        for k in all_infos.keys():
            all_infos[k].append(info[k])

    # 场景 C: 靠近终点 (1000步) - 强行将智能体放置在终点边缘
    for _ in range(1000):
        env.reset()
        # 将智能体放在距离终点 goal_tolerance + 0.2 的位置
        angle_to_goal = np.random.uniform(-np.pi, np.pi)
        dist = env.cfg.goal_tolerance + 0.2
        env.agent_pos = env.goal_pos + np.array([dist * np.cos(angle_to_goal), dist * np.sin(angle_to_goal)])
        env.agent_theta = angle_to_goal + np.pi  # 面向终点
        env.dist_to_goal_prev = np.linalg.norm(env.agent_pos - env.goal_pos)

        action = np.array([1.0, 0.0])  # 强制全速前进到达终点
        _, _, _, _, info = env.step(action)
        for k in all_infos.keys():
            all_infos[k].append(info[k])

    # 统计分析与格式化输出
    print(
        f"{'组件 (Component)':<18} | {'平均值':<8} | {'方差':<8} | {'最小值':<8} | {'25%':<8} | {'50%':<8} | {'75%':<8} | {'最大值':<8}")
    print("-" * 95)

    for key, values in all_infos.items():
        arr = np.array(values)
        mean_val = np.mean(arr)
        var_val = np.var(arr)
        min_val = np.min(arr)
        p25 = np.percentile(arr, 25)
        p50 = np.percentile(arr, 50)
        p75 = np.percentile(arr, 75)
        max_val = np.max(arr)

        print(
            f"{key:<18} | {mean_val:>8.4f} | {var_val:>8.4f} | {min_val:>8.4f} | {p25:>8.4f} | {p50:>8.4f} | {p75:>8.4f} | {max_val:>8.4f}")

    print("-" * 95)
    print("[+] 奖励组件分析完成！检查极值是否超出预设的 [-2, 2] 及极端惩罚/奖励范围。\n")


if __name__ == "__main__":
    # 初始化环境
    cfg = Config()
    env = Env(config=cfg)

    # 运行测试集
    test_spaces_and_interaction(env)
    test_generate_10_maps(env)
    test_reward_components(env)