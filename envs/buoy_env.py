# 模拟智能浮标交互环境

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
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
        
        # 环境状态初始化
        self.battery_capacity = 400  # Ah
        self.battery_level = self.battery_capacity  # 初始满电
        self.solar_panel_efficiency = 0.15  # 太阳能板效率
        self.solar_panel_size = 1  # m²
        # 日照
        self.solar_irradiance_range = {
            'Sunny': (600, 800),  # 晴天太阳辐照度范围（W/m²）
            'Cloudy': (300, 500),  # 多云天太阳辐照度范围（W/m²）
            'Rainy': (20, 80)     # 雨天太阳辐照度范围（W/m²）
        }
        self.solar_panel_size = 1  # 太阳能板大小为1平方米
        self.solar_panel_efficiency = 0.15  # 太阳能板效率为15%

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

        # 动态生成的属性
        self.monthly_sunlight_sequence = self.generate_monthly_sunlight_sequence()  # 预生成一个月的日照序列

        # 生成一个月的通信需求和消耗数据
        self.generate_monthly_communication_data()

        self.reset()  # 重置环境状态

    def generate_monthly_sunlight_sequence(self):
        np.random.seed(3)  # 设置随机种子以保持结果一致性，可以选择去掉以产生不同的序列
        self.weather_conditions = np.random.choice(['Sunny', 'Cloudy', 'Rainy'], 30, p=[0.5, 0.3, 0.2])  # 假设晴天的概率为50%，多云为30%，雨天为20%
                
        daily_solar_generation = []
        for condition in self.weather_conditions:
                irradiance_range = self.solar_irradiance_range[condition]
                irradiance = np.random.uniform(irradiance_range[0], irradiance_range[1])
                daily_generation = self.solar_panel_size * self.solar_panel_efficiency * irradiance
                daily_solar_generation.append(daily_generation)
                
        return daily_solar_generation  

    def generate_monthly_communication_data(self):
        # 生成一个月中每一天各传感器的能量消耗（单位：Wh）
        self.vhf_consumption_daily = np.random.uniform(5, 18, 30)  # VHF传感器每天的能量消耗
        self.ais_consumption_daily = np.random.uniform(1, 4, 30)  # AIS传感器每天的能量消耗
        
        # 4目摄像头在工作日和周末的能耗不同
        self.camera_consumption_weekday = np.random.uniform(18, 35, 30)  # 工作日
        self.camera_consumption_weekend = np.random.uniform(35, 65, 30)  # 周末
        for i in range(30):
            if (i+1) % 7 == 0 or (i+2) % 7 == 0:  # 周末
                self.camera_consumption_weekday[i] = self.camera_consumption_weekend[i]

        # 航标待机状态的基础能耗
        self.standby_consumption_daily = np.random.uniform(8, 12, 30)

        # 计算总能耗
        self.total_consumption_daily = self.vhf_consumption_daily + self.ais_consumption_daily + self.camera_consumption_weekday + self.standby_consumption_daily

        # 通信需求模型
        # 假设通信需求主要由AIS和VHF活动驱动，同时摄像头数据传输对通信需求有较大影响
        self.communication_demand_daily = (self.vhf_consumption_daily + self.ais_consumption_daily) * 1.2 + self.camera_consumption_weekday * 0.75

        # 根据通信需求调整因子，反映不同天气条件下的通信难度
        weather_impact = np.array([1.1 if condition == 'Sunny' else 1.3 if condition == 'Cloudy' else 1.5 for condition in self.weather_conditions])
        self.communication_demand_daily *= weather_impact

    def update_environment(self):
        # 计算当天的太阳能充电量
        solar_generation = self.monthly_sunlight_sequence[self.current_step]
        # 计算当天的总能量消耗
        total_consumption = self.calculate_energy_consumption()

        # 更新电池电量
        self.battery_level += solar_generation - total_consumption
        # 确保电池电量不会超过最大容量，也不会低于0
        self.battery_level = max(0, min(self.battery_level, self.battery_capacity))

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
        done = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), float(reward), done,truncated, {}

    def _get_obs(self):
        current_hour = (self.current_step % 24)
        current_day = (self.current_step % 7)
        return np.array([self.battery_level, self.communication_demand, self.channel_quality, self.current_step, current_hour, current_day, int(self.camera_on)])

    def reset(self):
        self.battery_level = self.battery_capacity
        self.current_step = 0
        self.lantern_on = False
        self.vhf_on = False
        self.camera_on = False
        self.radar_on = False
        self.qos_level = 0
        return self._get_obs()

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
        state, reward, done, info = env.step(action)
        print("State:", state, "Reward:", reward, "Done:", done)