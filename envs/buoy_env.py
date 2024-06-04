import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加当前文件所在目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .external_env import ExternalEnvironment

class SmartBuoyEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SmartBuoyEnvironment, self).__init__()
        
        # 定义状态空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 最小值
            high=np.array([400, 2, 100, 1, 23, 6, 1, 1, 1, 1, 3]),  # 最大值
            dtype=np.float32
        )
        
        # 定义动作空间
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 3])  # 航标灯、VHF、摄像头、毫米波雷达、通信QoS
        
        # 环境状态初始化
        self.battery_capacity = 400  # Ah
        self.battery_level = self.battery_capacity  # 初始满电
        self.solar_panel_efficiency = 0.15  # 太阳能板效率
        self.solar_panel_size = 1  # m²
        
        # 初始化外部环境
        self.external_env = ExternalEnvironment(seed=42)
        self.monthly_weather = self.external_env.simulate_monthly_weather()
        self.monthly_boat_traffic = self.external_env.simulate_monthly_boat_traffic()
        
        # 初始化其它状态
        self.communication_demand = 0  # 初始通信需求
        self.channel_quality = 0  # 初始通信信道质量
        self.current_step = 0
        self.max_steps = 30  # 每个周期的最大步数，比如一个月
        self.operational_days = 0  # 已操作的天数

        # 新增属性
        self.lantern_on = False
        self.vhf_on = False
        self.camera_on = False
        self.radar_on = False
        self.qos_level = 0
        self.ais_on = True  # AIS接收器实时工作
        self.edge_server_on = True  # 边缘服务器实时工作
        self.weather_monitor_on = True  # 水文气象监测仪实时工作

        self.reset()  # 重置环境状态

    def update_environment(self):
        # 获取当前天气
        current_weather = self.monthly_weather[self.current_step]
        # 计算当天的太阳能充电量
        solar_generation = self.external_env.get_solar_irradiance(current_weather)
        # 计算当天的总能量消耗
        total_consumption = self.calculate_energy_consumption()

        # 更新电池电量
        self.battery_level += solar_generation - total_consumption
        # 确保电池电量不会超过最大容量，也不会低于0
        self.battery_level = max(0, min(self.battery_level, self.battery_capacity))

        # 更新通信需求
        boat_count = self.monthly_boat_traffic[self.current_step]
        nearby_boats = self.external_env.monitor_boats(boat_count)
        self.communication_demand = self.external_env.simulate_communication_demand(nearby_boats)

        # 更新步骤
        self.current_step += 1

    def calculate_energy_consumption(self):
        energy_consumption = 0
        if self.lantern_on:
            energy_consumption += 5  # 假设灯笼每小时消耗5Wh
        if self.vhf_on:
            energy_consumption += 10  # 假设VHF每小时消耗10Wh
        if self.camera_on:
            energy_consumption += 15  # 假设摄像头每小时消耗15Wh
        if self.radar_on:
            energy_consumption += 20  # 假设雷达每小时消耗20Wh
        if self.ais_on:
            energy_consumption += 2  # 假设AIS每小时消耗2Wh
        if self.edge_server_on:
            energy_consumption += 5  # 假设边缘服务器每小时消耗5Wh
        if self.weather_monitor_on:
            energy_consumption += 1  # 假设水文气象监测仪每小时消耗1Wh
        return energy_consumption

    def step(self, action):
        # 解析动作
        self.lantern_on = action[0] == 1
        self.vhf_on = action[1] == 1
        self.camera_on = action[2] == 1
        self.radar_on = action[3] == 1
        self.qos_level = action[4]

        # 更新环境
        self.update_environment()

        # 计算奖励
        reward = -self.calculate_energy_consumption()
        if self.battery_level > 0:
            reward += 1  # 每小时电池电量大于0，奖励1分

        # 增加正向奖励，例如在完成特定任务时
        if self.communication_demand > 0 and self.qos_level > 1:
            reward += 10  # 假设在高QoS下满足通信需求时奖励10分

        done = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), float(reward), done, truncated, {}
    
    def _get_obs(self):
        current_hour = (self.current_step % 24)
        current_day = (self.current_step % 7)
        return np.array([
            self.battery_level, 
            self.communication_demand, 
            self.channel_quality, 
            self.current_step, 
            current_hour, 
            current_day, 
            int(self.lantern_on), 
            int(self.vhf_on), 
            int(self.camera_on), 
            int(self.radar_on), 
            self.qos_level
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.battery_level = self.battery_capacity
        self.current_step = 0
        self.lantern_on = False
        self.vhf_on = False
        self.camera_on = False
        self.radar_on = False
        self.qos_level = 0
        return self._get_obs(), {}

    def render(self, mode='human'):
        plt.figure(figsize=(10, 5))
        plt.title("Smart Buoy Environment")
        plt.bar(['Battery Level'], [self.battery_level], color='blue')
        plt.ylim(0, self.battery_capacity)
        plt.ylabel('Battery Level (Ah)')
        plt.xlabel('State')
        plt.show()

if __name__ == "__main__":
    env = SmartBuoyEnvironment()
    state = env.reset()
    print("Initial State:", state)
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print("State:", state, "Reward:", reward, "Done:", done)