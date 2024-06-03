import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment_changes import EnvironmentChanges

class SmartBuoyEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SmartBuoyEnvironment, self).__init__()
        
        # 定义动作空间的维度
        n = 3  # 通信功率和通信频率的状态数
        k = 3  # 能源消耗策略的状态数
        m = 2  # 传感器开关状态（开/关）
        
        self.n = n
        self.k = k
        self.m = m
        
        # 将MultiDiscrete动作空间转换为Discrete动作空间
        self.action_space = spaces.Discrete(n * n * k * m * m * m)  # 定义动作空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),  # 最小值
            high=np.array([400, 2, 100, 1, 23, 6, 1]),  # 最大值，假设时间为0-23小时，日期为0-6（周一到周日），传感器模式为0-1
            dtype=np.float32
        )
        
        # 环境状态初始化
        self.mqtt_qos_levels = [0, 1, 2]  # 定义MQTT QoS 级别
        self.video_transmission_modes = ["LowQuality", "StandardQuality", "HighQuality"]  # 定义视频传输模式列表
        self.communication_power_levels = [0.5, 1.0, 1.5]  # 通信功率等级
        self.communication_frequency_levels = [900, 1800, 2600]  # 通信频率等级

        self.battery_capacity = 400  # Ah
        self.battery_level = self.battery_capacity  # 初始满电
        self.solar_panel_efficiency = 0.15  # 太阳能板效率
        self.solar_panel_size = 1  # m²

        # 初始化其它状态
        self.communication_demand = 0  # 初始通信需求
        self.channel_quality = 0  # 初始通信信道质量
        self.current_step = 0
        self.max_steps = 30  # 每个周期的最大步数，比如一个月
        self.operational_days = 0  # 已操作的天数
        self.sensor_mode = 0  # 传感器模式，0表示Normal，1表示LowPower

        # 动态生成的属性
        self.env_changes = EnvironmentChanges()
        self.monthly_weather, self.monthly_boat_counts = self.env_changes.simulate_environment(0)

        # 将天气条件转换为相应的辐照度值
        self.weather_to_irradiance()

        # 生成一个月的通信需求和消耗数据
        self.generate_monthly_communication_data()

        self.reset()  # 重置环境状态

    def weather_to_irradiance(self):
        irradiance_mapping = {
            'Sunny': np.random.uniform(600, 800),
            'Cloudy': np.random.uniform(300, 500),
            'Rainy': np.random.uniform(20, 80)
        }
        self.monthly_weather = [irradiance_mapping[condition] for condition in self.monthly_weather]

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
        weather_impact = np.array([1.1 if condition == 'Sunny' else 1.3 if condition == 'Cloudy' else 1.5 for condition in self.monthly_weather])
        self.communication_demand_daily *= weather_impact

    def update_environment(self):
        # 根据当前天气更新太阳能发电量
        if self.current_step < len(self.monthly_weather):
            day_index = self.current_step
            solar_generation = self.solar_panel_size * self.solar_panel_efficiency * self.monthly_weather[day_index]
        else:
            solar_generation = 0  # 超出预设日照序列范围时的处理

        # 更新蓄电池电量，考虑能量消耗和太阳能发电
        self.battery_level += solar_generation - self.total_consumption_daily[self.current_step]
        self.battery_level = max(0, min(self.battery_level, self.battery_capacity))  # 保证电量在有效范围内

        # 更新通信需求
        if self.current_step < len(self.communication_demand_daily):
            self.communication_demand = self.communication_demand_daily[self.current_step]
        else:
            self.communication_demand = 0  # 超出预设范围时的处理

        # 动态调整传感器的工作模式
        self.adjust_sensor_mode()

        # 更新当前步数
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.current_step = 0  # 或其他逻辑，如终止环境等

    def adjust_sensor_mode(self):
        if self.battery_level < 100:
            self.sensor_mode = 1  # LowPower模式
            # 关闭部分摄像头
            self.camera_consumption_weekday *= 0.5
            self.camera_consumption_weekend *= 0.5
        else:
            self.sensor_mode = 0  # Normal模式

    def apply_action(self, action):
        # 将Discrete动作空间转换为MultiDiscrete动作空间
        power_index = action % self.n
        frequency_index = (action // self.n) % self.n
        strategy_index = (action // (self.n * self.n)) % self.k
        vhf_on = (action // (self.n * self.n * self.k)) % self.m
        ais_on = (action // (self.n * self.n * self.k * self.m)) % self.m
        camera_on = (action // (self.n * self.n * self.k * self.m * self.m)) % self.m

        self.communication_power = self.communication_power_levels[power_index]
        self.communication_frequency = self.communication_frequency_levels[frequency_index]
        self.energy_strategy = strategy_index

        # 控制传感器开关
        self.vhf_on = vhf_on
        self.ais_on = ais_on
        self.camera_on = camera_on

        # 假设通信成功率与通信功率和频率相关
        self.communication_success_rate = self.calculate_communication_success_rate(power_index, frequency_index)

        print(f"Action applied: Power index={power_index}, Frequency index={frequency_index}, Strategy index={strategy_index}, VHF on={vhf_on}, AIS on={ais_on}, Camera on={camera_on}")

    def calculate_communication_success_rate(self, power_index, frequency_index):
        # 这里是一个示例逻辑，您应该根据项目的实际情况来调整
        success_rate = 0.5 + 0.1 * power_index + 0.05 * frequency_index
        return min(1.0, success_rate)  # 确保成功率不超过1

    def get_state(self):
        # 将天气状况从字符串映射到数值
        weather_mapping = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
        weather_condition_numeric = weather_mapping.get(self.monthly_weather[self.current_step], 0)

        # 获取当前的通信需求和通信信道质量
        communication_demand = self.communication_demand
        channel_quality = self.channel_quality

        # 组合成一个观测向量
        state = np.array([self.battery_level, weather_condition_numeric, communication_demand, channel_quality, self.current_step % 24, self.current_step % 7, self.sensor_mode])
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery_level = self.battery_capacity
        self.monthly_weather, self.monthly_boat_counts = self.env_changes.simulate_environment(0)
        self.weather_to_irradiance()  # 将天气条件转换为相应的辐照度值
        self.generate_monthly_communication_data()
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

    def calculate_reward(self):
        # 这里是一个示例逻辑，您应该根据项目的实际情况来调整
        reward = self.communication_success_rate - (self.communication_demand / 100)
        if self.battery_level < 50:
            reward -= 1  # 电池电量过低时，减少奖励
        if self.sensor_mode == 1:  # LowPower模式
            reward += 0.5  # 低功耗模式下，增加奖励
        return reward