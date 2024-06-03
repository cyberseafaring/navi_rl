# 模拟智能浮标交互环境

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .external_env import ExternalEnvironment

class SmartBuoyEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SmartBuoyEnvironment, self).__init__()
        
        # 定义状态空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),  # 最小值
            high=np.array([400, 2, 100, 1, 23, 6, 1]),  # 最大值，假设时间为0-23小时，日期为0-6（周一到周日），传感器模式为0-1
            dtype=np.float32
        )
        
        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 3])  # 航标灯、VHF、摄像头、毫米波雷达、通信QoS
        
        # 初始化外部环境
        self.external_env = ExternalEnvironment(seed=42)
        self.monthly_weather = self.external_env.simulate_monthly_weather()
        self.monthly_boat_traffic = self.external_env.simulate_monthly_boat_traffic()
        self.daily_time = self.external_env.simulate_daily_time()
        
        # 初始化内部状态
        self.battery_capacity = 400  # Ah
        self.battery_level = self.battery_capacity  # 初始满电
        self.solar_panel_efficiency = 0.15  # 太阳能板效率
        self.solar_panel_size = 1  # m²
        self.current_step = 0
        self.max_steps = 30  # 每个周期的最大步数，比如一个月
        self.sensor_mode = 0  # 传感器模式，0表示Normal，1表示LowPower

        self.reset()  # 重置环境状态

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery_level = self.battery_capacity
        self.monthly_weather = self.external_env.simulate_monthly_weather()
        self.monthly_boat_traffic = self.external_env.simulate_monthly_boat_traffic()
        self.daily_time = self.external_env.simulate_daily_time()
        return self.get_state(), {}

    def step(self, action):
        self.apply_action(action)
        self.update_environment()
        state = self.get_state()
        reward = self.calculate_reward()
        done = self.current_step == 0
        truncated = False  # 你可以根据需要设置截断条件
        info = {}
        return state, reward, done, truncated, info

    def apply_action(self, action):
        self.lantern_on, self.vhf_on, self.camera_on, self.radar_on, self.qos_level = action

        # 控制航标灯
        if self.daily_time[self.current_step % 24] < 6 or self.daily_time[self.current_step % 24] >= 18:
            self.lantern_on = 1  # 夜晚开启
        else:
            self.lantern_on = 0  # 白天关闭

        print(f"Action applied: Lantern on={self.lantern_on}, VHF on={self.vhf_on}, Camera on={self.camera_on}, Radar on={self.radar_on}, QoS level={self.qos_level}")

    def update_environment(self):
        # 根据当前天气更新太阳能发电量
        day_index = self.current_step % 30
        solar_generation = self.solar_panel_size * self.solar_panel_efficiency * self.external_env.get_solar_irradiance(self.monthly_weather[day_index])

        # 更新蓄电池电量，考虑能量消耗和太阳能发电
        self.battery_level += solar_generation - self.calculate_energy_consumption()
        self.battery_level = max(0, min(self.battery_level, self.battery_capacity))  # 保证电量在有效范围内

        # 更新当前步数
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.current_step = 0  # 或其他逻辑，如终止环境等

    def calculate_energy_consumption(self):
        # 计算能耗
        energy_consumption = 0
        if self.lantern_on:
            energy_consumption += 5  # 航标灯能耗
        if self.vhf_on:
            energy_consumption += 10  # VHF能耗
        if self.camera_on:
            energy_consumption += 15  # 摄像头能耗
        if self.radar_on:
            energy_consumption += 20  # 毫米波雷达能耗
        energy_consumption += self.qos_level * 5  # 通信QoS能耗
        return energy_consumption

    def get_state(self):
        # 将天气状况从字符串映射到数值
        weather_mapping = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
        weather_condition_numeric = weather_mapping.get(self.monthly_weather[self.current_step % 30], 0)

        # 获取当前的通信需求和通信信道质量
        communication_demand = self.monthly_boat_traffic[self.current_step % 30]
        channel_quality = 1  # 假设信道质量为1

        # 组合成一个观测向量
        state = np.array([self.battery_level, weather_condition_numeric, communication_demand, channel_quality, self.current_step % 24, self.current_step % 7, self.sensor_mode])
        return state

    def calculate_reward(self):
        # 这里是一个示例逻辑，您应该根据项目的实际情况来调整
        reward = -self.calculate_energy_consumption()  # 能耗越低，奖励越高
        if self.battery_level < 50:
            reward -= 10  # 电池电量过低时，减少奖励
        if self.sensor_mode == 1:  # LowPower模式
            reward += 5  # 低功耗模式下，增加奖励
        return reward

# 示例用法
if __name__ == "__main__":
    env = SmartBuoyEnvironment()
    state, _ = env.reset()
    print("Initial State:", state)
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print("State:", state, "Reward:", reward, "Done:", done)