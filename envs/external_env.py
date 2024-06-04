import numpy as np

class ExternalEnvironment:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # 定义天气条件
        self.weather_conditions = ['Sunny', 'Cloudy', 'Rainy']
        self.weather_probabilities = [0.5, 0.3, 0.2]  # 晴天50%，多云30%，雨天20%
        
        # 定义船只流量
        self.weekday_boat_range = (5, 15)  # 工作日船只流量范围
        self.weekend_boat_range = (10, 25)  # 周末船只流量范围
        
        # 定义时间变化
        self.day_hours = 12  # 假设白天12小时
        self.night_hours = 24 - self.day_hours

    def simulate_monthly_weather(self):
        # 模拟一个月的天气变化
        monthly_weather = np.random.choice(self.weather_conditions, 30, p=self.weather_probabilities)
        return monthly_weather

    def simulate_monthly_boat_traffic(self):
        # 模拟一个月的船只流量
        monthly_boat_traffic = []
        for day in range(30):
            if day % 7 == 5 or day % 7 == 6:  # 周末
                boat_count = np.random.randint(self.weekend_boat_range[0], self.weekend_boat_range[1])
            else:  # 工作日
                boat_count = np.random.randint(self.weekday_boat_range[0], self.weekday_boat_range[1])
            monthly_boat_traffic.append(boat_count)
        return monthly_boat_traffic

    def simulate_daily_time(self):
        # 模拟一天的时间变化，区分白天和夜晚
        daily_time = np.arange(24)
        return daily_time

    def get_solar_irradiance(self, weather):
        # 根据天气条件返回相应的太阳能辐照度
        irradiance_mapping = {
            'Sunny': np.random.uniform(600, 800),
            'Cloudy': np.random.uniform(300, 500),
            'Rainy': np.random.uniform(20, 80)
        }
        return irradiance_mapping[weather]

    def get_energy_consumption(self, hour):
        # 根据时间返回相应的能耗
        if 6 <= hour < 18:  # 白天
            return 'day'
        else:  # 夜晚
            return 'night'

    def monitor_boats(self, boat_distance):
        # 监测船只，如果船只在500米以内则需要主动监测
        if boat_distance <= 500:
            return True  # 需要监测
        return False  # 不需要监测

# 示例用法
if __name__ == "__main__":
    env = ExternalEnvironment(seed=42)
    monthly_weather = env.simulate_monthly_weather()
    monthly_boat_traffic = env.simulate_monthly_boat_traffic()
    daily_time = env.simulate_daily_time()

    print("Monthly Weather:", monthly_weather)
    print("Monthly Boat Traffic:", monthly_boat_traffic)
    print("Daily Time:", daily_time)