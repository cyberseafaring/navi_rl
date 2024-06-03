import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from env import SmartBuoyEnvironment
import numpy as np

# 创建向量化环境
env = make_vec_env(lambda: SmartBuoyEnvironment(), n_envs=1)

# 创建DQN模型
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-4,  # 调整学习率
    buffer_size=50000,  # 增加经验回放缓冲区大小
    learning_starts=1000, 
    batch_size=64,  # 增加批量大小
    tau=1.0, 
    gamma=0.99, 
    train_freq=4, 
    gradient_steps=1, 
    target_update_interval=500,  # 增加目标网络更新间隔
    exploration_fraction=0.1, 
    exploration_initial_eps=1.0, 
    exploration_final_eps=0.01,  # 减少最终探索率
    max_grad_norm=10, 
    tensorboard_log="./dqn_smartbuoy_tensorboard/"
)

# 训练模型
model.learn(total_timesteps=100000)

# 保存模型
model.save("dqn_smartbuoy")

# 定义基线策略
def random_policy(env):
    obs = env.reset()
    done = [False]
    days = 0
    while not done[0]:
        action = [env.action_space.sample()]  # 随机选择动作
        obs, reward, done, info = env.step(action)
        days += 1
    return days

# 使用DQN算法的策略
def dqn_policy(env, model):
    obs = env.reset()
    done = [False]
    days = 0
    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        days += 1
    return days

# 创建环境实例
env = make_vec_env(lambda: SmartBuoyEnvironment(), n_envs=1)

# 加载训练好的DQN模型
model = DQN.load("dqn_smartbuoy")

# 比较两种策略的运行天数
dqn_days = dqn_policy(env, model)
random_days = random_policy(env)

print(f"DQN策略运行天数: {dqn_days}")
print(f"随机策略运行天数: {random_days}")