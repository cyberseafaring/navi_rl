import numpy as np

class ExternalEnvironment:
    def __init__(self, seed=None):
        self.seed = seed
        self.reset_seed()

        # 定义天气条件
        self.weather_conditions = ['Sunny', 'Cloudy', 'Rainy']
        self.weather_probabilities = [0.5, 0.3, 0.2]  # 晴天50%，多云30%，雨天20%
        
        # 定义船只流量
        self.weekday_boat_range = (250, 350)  # 工作日船只流量范围
        self.weekend_boat_range = (310, 425)  # 周末船只流量范围
        
        # 定义时间变化
        self.day_hours = 12  # 假设白天12小时
        self.night_hours = 24 - self.day_hours

    def reset_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            self.seed += 1  # 每次重置时更新随机种子

    def simulate_monthly_weather(self):
        self.reset_seed()
        monthly_weather = np.random.choice(self.weather_conditions, 30, p=self.weather_probabilities)
        return monthly_weather

    def simulate_monthly_boat_traffic(self):
        self.reset_seed()
        monthly_boat_traffic = []
        for day in range(30):
            if day % 7 == 5 or day % 7 == 6:  # 周末
                boat_count = np.random.randint(self.weekend_boat_range[0], self.weekend_boat_range[1])
            else:  # 工作日
                boat_count = np.random.randint(self.weekday_boat_range[0], self.weekday_boat_range[1])
            monthly_boat_traffic.append(boat_count)
        return monthly_boat_traffic

    def simulate_daily_time(self):
        self.reset_seed()
        daily_time = np.arange(24)
        return daily_time

    def get_solar_irradiance(self, weather):
        self.reset_seed()
        irradiance_mapping = {
            'Sunny': np.random.uniform(600, 800),
            'Cloudy': np.random.uniform(300, 500),
            'Rainy': np.random.uniform(20, 80)
        }
        return irradiance_mapping[weather]

    def get_energy_consumption(self, hour):
        if 6 <= hour < 18:  # 白天
            return 'day'
        else:  # 夜晚
            return 'night'

    def monitor_boats(self, boat_count):
        self.reset_seed()
        # 假设船只在浮标500米以内的概率为10%
        nearby_boats = np.random.binomial(boat_count, 0.1)
        # 确保同一时间段在浮标附近的船只数量不超过3艘
        return min(nearby_boats, 3)

# 示例用法
if __name__ == "__main__":
    env = ExternalEnvironment(seed=11)
    monthly_weather = env.simulate_monthly_weather()
    monthly_boat_traffic = env.simulate_monthly_boat_traffic()
    daily_time = env.simulate_daily_time()

    print("Monthly Weather:", monthly_weather)
    print("Monthly Boat Traffic:", monthly_boat_traffic)
    print("Daily Time:", daily_time)

    # 模拟某一天的船只监测
    boat_count = monthly_boat_traffic[0]
    nearby_boats = env.monitor_boats(boat_count)
    print("Nearby Boats:", nearby_boats)