import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from env import Env, Config, Plot


# ==========================================
# 1. PPO 网络结构定义定义 (Actor-Critic MLP)
# ==========================================
def orthogonal_init(layer, gain=1.0):
    """正交初始化，PPO在连续控制任务中的标准操作"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Actor 网络 (输出连续动作的均值)
        self.actor_mean = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)  # 最后一层小方差初始化
        )
        # Actor 的对数标准差 (独立于状态的可学习参数)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic 网络 (输出状态价值 V)
        self.critic = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)
        )

    def forward(self):
        raise NotImplementedError

    def get_action_and_value(self, state, action=None):
        action_mean = self.actor_mean(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        # PPO 要求的对数概率、熵和状态价值
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(state).squeeze(-1)


# ==========================================
# 2. PPO 算法核心逻辑
# ==========================================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def update(self, rollouts, clip_vloss=True):
        b_states, b_actions, b_logprobs, b_returns, b_advantages, b_values = rollouts

        # 将优势函数标准化 (加速收敛)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 计算新策略下的输出
        _, newlogprob, entropy, newvalue = self.network.get_action_and_value(b_states, b_actions)
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        # Actor 损失 (Clipping 目标)
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Critic 价值损失
        if clip_vloss:
            v_loss_unclipped = (newvalue - b_returns) ** 2
            v_clipped = b_values + torch.clamp(newvalue - b_values, -self.clip_coef, self.clip_coef)
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

        # 熵奖励 (鼓励探索)
        entropy_loss = entropy.mean()

        # 总损失
        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

        # 梯度反向传播与裁剪
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item(), pg_loss.item(), v_loss.item()


# ==========================================
# 3. 训练主循环与数据记录
# ==========================================
def train():
    os.makedirs('models', exist_ok=True)

    # 训练配置
    cfg = Config()
    env = Env(config=cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # PPO 超参数
    total_timesteps = 1_000_000  # 总交互步数
    num_steps = 2048  # 每次收集的步数 (Rollout length)
    ppo_epochs = 10  # 每次收集后网络更新的 epoch 数
    batch_size = 64  # 更新时的 mini-batch 大小

    agent = PPOAgent(state_dim, action_dim)

    # 学习率调度器 (线性衰减)
    num_updates = total_timesteps // num_steps
    scheduler = optim.lr_scheduler.LinearLR(agent.optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_updates)

    # 日志变量
    global_step = 0
    episode_rewards = []
    plot_rewards = []
    success_count = 0
    episodes_in_log_window = 0
    ep_lens = []

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(agent.device)

    print(f"开始训练 PPO: 状态维度 {state_dim}, 动作维度 {action_dim}")
    print("=" * 60)

    for update in range(1, num_updates + 1):
        # 初始化缓冲区
        obs_buf, acts_buf, logprobs_buf, rewards_buf, vals_buf, dones_buf = [], [], [], [], [], []

        # 1. 收集数据 (Rollout)
        agent.network.eval()
        for step in range(num_steps):
            global_step += 1

            with torch.no_grad():
                action, logprob, _, value = agent.network.get_action_and_value(state.unsqueeze(0))

            # 存储转换数据
            obs_buf.append(state)
            acts_buf.append(action.squeeze(0))
            logprobs_buf.append(logprob.squeeze(0))
            vals_buf.append(value.squeeze(0))

            # 环境交互
            action_np = action.squeeze(0).cpu().numpy()
            next_state, reward, done, truncated, info = env.step(action_np)

            rewards_buf.append(torch.tensor(reward, dtype=torch.float32).to(agent.device))
            dones_buf.append(torch.tensor(done, dtype=torch.float32).to(agent.device))

            state = torch.tensor(next_state, dtype=torch.float32).to(agent.device)

            if done or truncated:
                ep_reward = sum([r.item() for r in rewards_buf[-env.steps:]]) if env.steps <= len(rewards_buf) else 0
                # 记录回合数据
                if 'r_goal' in info and info['r_goal'] > 0:
                    success_count += 1

                episodes_in_log_window += 1
                ep_lens.append(env.steps)

                # 为了后续绘图，仅在真实回合结束时记录真实总奖励
                # (这里简化计算，实际中通常有专门的 Wrapper 处理 episodic return)
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32).to(agent.device)

        # 2. 计算 GAE (广义优势估计)
        with torch.no_grad():
            next_value = agent.network.get_action_and_value(state.unsqueeze(0))[3].squeeze(0)
            advantages = torch.zeros_like(torch.stack(rewards_buf)).to(agent.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = vals_buf[t + 1]
                delta = rewards_buf[t] + agent.gamma * nextvalues * nextnonterminal - vals_buf[t]
                advantages[t] = lastgaelam = delta + agent.gamma * agent.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + torch.stack(vals_buf)

        # 展平 Tensor 供网络更新使用
        b_obs = torch.stack(obs_buf)
        b_acts = torch.stack(acts_buf)
        b_logprobs = torch.stack(logprobs_buf)
        b_returns = returns
        b_advantages = advantages
        b_vals = torch.stack(vals_buf)

        # 3. 更新网络 (PPO Epochs)
        agent.network.train()
        b_inds = np.arange(num_steps)
        clip_fracs = []
        total_loss, total_pg_loss, total_v_loss = 0, 0, 0

        for epoch in range(ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                loss, pg_loss, v_loss = agent.update(
                    (b_obs[mb_inds], b_acts[mb_inds], b_logprobs[mb_inds],
                     b_returns[mb_inds], b_advantages[mb_inds], b_vals[mb_inds])
                )
                total_loss += loss
                total_pg_loss += pg_loss
                total_v_loss += v_loss

        scheduler.step()  # 更新学习率

        # 4. 打印训练信息与保存模型
        if update % 5 == 0:
            avg_len = np.mean(ep_lens) if len(ep_lens) > 0 else 0
            success_rate = (success_count / episodes_in_log_window) * 100 if episodes_in_log_window > 0 else 0
            current_lr = scheduler.get_last_lr()[0]

            # 计算这一批次的平均奖励作为绘图点
            batch_avg_reward = sum(rewards_buf).item() / episodes_in_log_window if episodes_in_log_window > 0 else 0
            plot_rewards.append(batch_avg_reward)

            print(f"Update: {update}/{num_updates} | Steps: {global_step} | "
                  f"Avg Len: {avg_len:.1f} | Success Rate: {success_rate:.1f}% | "
                  f"Loss: {total_loss / (ppo_epochs * (num_steps / batch_size)):.4f} | LR: {current_lr:.6f}")

            # 重置日志窗口变量
            success_count = 0
            episodes_in_log_window = 0
            ep_lens = []

        # 每 50 次更新保存一次模型
        if update % 50 == 0:
            torch.save(agent.network.state_dict(), f"models/ppo_nav_step_{global_step}.pth")

    print("=" * 60)
    print("训练结束！正在绘制训练曲线...")

    # 5. 调用绘图接口
    Plot.plot_learning_curve(plot_rewards, window=10)


if __name__ == "__main__":
    train()