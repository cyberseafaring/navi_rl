import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SmartBuoyEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        #print("Initializing environment...")
        super(SmartBuoyEnvironment, self).__init__()
        
        # 定义动作空间的维度
        n = 3  # 通信功率和通信频率的状态数
        k = 3  # 能源消耗策略的状态数
        
        self.action_space = spaces.Discrete(n * n * k)  # 定义动作空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # 最小值
            high=np.array([400, 2, 100, 1]),  # 最大值，假设通信需求最大为100，天气条件用0到2表示，信道质量用0到1表示
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
        # 根据当前天气更新太阳能发电量
        if self.current_day <= len(self.monthly_sunlight_sequence):
                day_index = self.current_day - 1
                solar_generation = self.solar_panel_size * self.solar_panel_efficiency * self.monthly_sunlight_sequence[day_index]
        else:
                solar_generation = 0  # 超出预设日照序列范围时的处理

        # 更新蓄电池电量，考虑能量消耗和太阳能发电
        self.battery_level += solar_generation - self.total_consumption_daily[self.current_day - 1]
        self.battery_level = max(0, min(self.battery_level, self.battery_capacity))  # 保证电量在有效范围内

        # 示例：更新通信需求
        # 这里假设已经在generate_monthly_communication_data中生成了每日的通信需求
        if self.current_day <= len(self.communication_demand_daily):
                self.communication_demand = self.communication_demand_daily[self.current_day - 1]
        else:
                self.communication_demand = 0  # 超出预设范围时的处理

        # 根据需要添加更多环境状态更新的逻辑

        # 更新当前天数
        self.current_day += 1
        if self.current_day > self.max_steps:
                self.current_day = 1  # 或其他逻辑，如终止环境等

    def apply_action(self, action):
        n = 3  # 通信功率和通信频率的状态数
        k = 3  # 能源消耗策略的状态数

        power_index = action % n
        frequency_index = (action // n) % n
        strategy_index = action // (n * n)

        self.communication_power = self.communication_power_levels[power_index]
        self.communication_frequency = self.communication_frequency_levels[frequency_index]
        self.energy_strategy = strategy_index

        # 假设通信成功率与通信功率和频率相关
        self.communication_success_rate = self.calculate_communication_success_rate(power_index, frequency_index)

        print(f"Action applied: Power index={power_index}, Frequency index={frequency_index}, Strategy index={strategy_index}")

    def calculate_communication_success_rate(self, power_index, frequency_index):
        # 这里是一个示例逻辑，您应该根据项目的实际情况来调整
        success_rate = 0.5 + 0.1 * power_index + 0.05 * frequency_index
        return min(1.0, success_rate)  # 确保成功率不超过1



    def get_state(self):
        # 将天气状况从字符串映射到数值
        weather_mapping = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
        weather_condition_numeric = weather_mapping.get(self.weather_conditions[self.current_day - 1], 0)

        # 获取当前的通信需求和通信信道质量
        # 注意这里简化处理，直接取当前天的通信需求和信道质量，实际应用中可能需要更复杂的逻辑
        communication_demand = self.communication_demand
        channel_quality = self.channel_quality

        # 组合成一个观测向量
        state = np.array([self.battery_level, weather_condition_numeric, communication_demand, channel_quality], dtype=np.float32)
        
        return state
    
    def calculate_reward(self):
        # 基础奖励
        reward = 0

        # 添加基于`communication_success_rate`的奖励逻辑
        reward += self.communication_success_rate * 100  # 假设每增加1%的成功率，奖励增加100点
        
        # 1. 能源效率奖励：电池剩余比例
        battery_ratio = self.battery_level / self.battery_capacity
        reward_battery = battery_ratio * 100  # 示例：电池比例每增加1%，奖励增加100点

        # 2. 通信需求满足度奖励：根据通信需求和实际消耗进行计算
        # 假设self.communication_success_rate是通信成功的比例，范围[0,1]
        reward_communication = self.communication_success_rate * 100  # 示例：通信成功率每增加1%，奖励增加100点

        # 3. 惩罚措施：电量耗尽或通信需求未能完全满足
        penalty_battery_depletion = -1000 if self.battery_level <= 0 else 0  # 电池耗尽惩罚
        penalty_communication_failure = -500 * (1 - self.communication_success_rate)  # 通信失败惩罚

        
        # 计算总奖励
        total_reward = reward + reward_battery + reward_communication + penalty_battery_depletion + penalty_communication_failure
        
        return total_reward


    def step(self, action):
        #print(f"Taking step with action: {action}")
        # 根据提供的动作更新环境状态
        self.apply_action(action)
        self.update_environment()

        # 计算奖励和判断episode是否结束
        reward = self.calculate_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = {
                "battery_level": self.battery_level,
                "day": self.current_day,
                "operational_days": self.operational_days
        }

        observation = self.get_state()
        
        # 更新步骤计数器和当前日期
        self.current_step += 1
        if self.current_day >= self.max_steps:
                truncated = True

        # 检查是否达到环境的终止条件
        if self.battery_level <= 0:
                terminated = True

        return observation, reward, terminated, truncated, info

    def is_terminated(self):
        # 检查是否达到终止条件，例如电量耗尽
        return self.battery_level <= 0

    def is_truncated(self):
        # 检查是否因为达到最大步数等原因导致截断
        return self.current_step >= self.max_steps

    def reset(self, **kwargs):
        #print("Resetting environment...")
        if 'seed' in kwargs:
            seed = kwargs['seed']
        # 重置蓄电池电量到满电
        self.battery_level = self.battery_capacity
        
        # 重置当前日期到第一天
        self.current_day = 1
        
        # 重置通信需求和通信信道质量到初始值
        self.communication_demand = 0
        self.channel_quality = 0
        
        # 重新生成一个月的日照序列
        self.monthly_sunlight_sequence = self.generate_monthly_sunlight_sequence()
        
        # 重置记录蓄电池电量的变化列表
        self.battery_levels = []

        # 重置服务提供状态
        self.operational_days = 0
        self.services_provided = {'VHF': 0, 'AIS': 0, 'Video_Transmission': 0}

        # 重置其他相关状态
        self.current_step = 0
        self.video_transmission_mode = self.video_transmission_modes[1]  # 默认设置为"StandardQuality"
        self.mqtt_qos = self.mqtt_qos_levels[1]  # 默认设置为 QoS 1
        self.communication_power = self.communication_power_levels[1]  # 默认功率设置为中等
        self.communication_frequency = self.communication_frequency_levels[1]  # 默认频率设置为中等

        # 更新环境以反映重置状态
        self.update_environment()

        # 返回初始观测状态
        return self.get_state(), {}


    def render(self, mode='human', close=False):
        if close:
            # 如果需要的话，在这里添加清理渲染资源的逻辑
            print("Closing the rendering")
            return

        if mode == 'human':
            print(f"Day {self.current_day}/{self.max_steps}:")
            print(f"  Battery level: {self.battery_level:.2f} Ah")
            print(f"  Weather condition: {self.weather_condition}")
            print(f"  Solar generation today: {self.solar_panel_size * self.monthly_sunlight_sequence[self.current_day-1] * self.solar_panel_efficiency:.2f} Wh")
            print(f"  Communication demand: {self.communication_demand:.2f}")
            print(f"  Channel quality: {self.channel_quality:.2f}")
            print(f"  Operational days: {self.operational_days}")
            services = ", ".join([f"{service}: {count}" for service, count in self.services_provided.items()])
            print(f"  Services provided: {services}")

    def close(self):
        # 如果有开启的资源，比如图形界面、文件等，在这里进行关闭
        print("Closing environment and releasing resources")
        # 示例：关闭图形界面
        # if self.some_graphical_interface:
        #     self.some_graphical_interface.close()
        
        # 示例：如果有网络连接或外部设备连接，在这里断开
        # if self.some_external_connection:
        #     self.some_external_connection.disconnect()

