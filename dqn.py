import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 假设你的环境定义在env.py文件中，确保SmartBuoyEnvironment类已经被正确定义
from env import SmartBuoyEnvironment

# 由于Stable Baselines3推荐使用向量化环境，我们将环境包装成向量化环境
# 这里我们只使用一个环境实例进行训练
env = make_vec_env(lambda: SmartBuoyEnvironment(), n_envs=1)

# 创建DQN模型
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=10000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=250, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10, tensorboard_log="./dqn_smartbuoy_tensorboard/")

# 训练模型
model.learn(total_timesteps=50000)

# 保存模型
model.save("dqn_smartbuoy")

# 重新加载模型（可选）
model = DQN.load("dqn_smartbuoy")

# 评估模型
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward}")
# 直接创建一个新的环境实例进行评估
test_env = make_vec_env(lambda: SmartBuoyEnvironment(), n_envs=1)
mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# 测试模型
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        env.reset()

# 关闭环境
env.close()

# 使用`tensorboard --logdir=./dqn_smartbuoy_tensorboard/`命令查看训练结果。