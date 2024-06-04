import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入自定义环境
from envs.buoy_env import SmartBuoyEnvironment

def train_ppo_agent(env_id, total_timesteps=100000, log_dir="logs", save_dir="models"):
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 创建环境
    env = make_vec_env(env_id, n_envs=1, vec_env_cls=DummyVecEnv)

    # 创建PPO模型
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # 创建回调函数，用于保存模型和评估
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_dir, name_prefix="ppo_model")
    eval_callback = EvalCallback(env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=500, deterministic=True, render=False)

    # 训练模型
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    # 保存最终模型
    model.save(os.path.join(save_dir, "ppo_final_model"))

    return model

def evaluate_ppo_agent(model_path, env_id, n_eval_episodes=10):
    # 创建环境
    env = make_vec_env(env_id, n_envs=1, vec_env_cls=DummyVecEnv)

    # 加载模型
    model = PPO.load(model_path)

    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    env_id = "SmartBuoyEnvironment-v0"

    # 注册自定义环境
    gym.envs.register(
        id=env_id,
        entry_point='envs.buoy_env:SmartBuoyEnvironment',
    )

    # 训练PPO智能体
    model = train_ppo_agent(env_id)

    # 评估PPO智能体
    evaluate_ppo_agent(os.path.join("models", "ppo_final_model.zip"), env_id)