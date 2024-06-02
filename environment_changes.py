import numpy as np

class EnvironmentChanges:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.days_in_month = 30
        self.weather_conditions = ['Sunny', 'Cloudy', 'Rainy']
        self.weather_probabilities = [0.5, 0.3, 0.2]
        self.weekend_boat_increase_factor = 1.5
        self.base_boat_count = 10
        self.boat_count_variation = 5

    def generate_monthly_weather(self):
        # 生成一个月的天气变化
        monthly_weather = np.random.choice(self.weather_conditions, self.days_in_month, p=self.weather_probabilities)
        return monthly_weather

    def generate_monthly_boat_counts(self):
        # 生成一个月的船只数量变化
        monthly_boat_counts = []
        for day in range(self.days_in_month):
            if (day + 1) % 7 == 0 or (day + 2) % 7 == 0:  # 周末
                boat_count = self.base_boat_count * self.weekend_boat_increase_factor
            else:
                boat_count = self.base_boat_count
            boat_count += np.random.randint(-self.boat_count_variation, self.boat_count_variation)
            monthly_boat_counts.append(max(0, boat_count))  # 确保船只数量不为负数
        return monthly_boat_counts

    def adjust_hyperparameters(self, batch_number):
        # 根据批次调整超参数
        self.weather_probabilities = [0.5 - 0.01 * batch_number, 0.3, 0.2 + 0.01 * batch_number]
        self.weekend_boat_increase_factor = 1.5 + 0.05 * batch_number

    def simulate_environment(self, batch_number):
        # 调整超参数
        self.adjust_hyperparameters(batch_number)
        
        # 生成环境变化
        monthly_weather = self.generate_monthly_weather()
        monthly_boat_counts = self.generate_monthly_boat_counts()
        
        return monthly_weather, monthly_boat_counts

if __name__ == "__main__":
    env_changes = EnvironmentChanges()
    
    # 假设我们要模拟10个批次
    for batch in range(10):
        weather, boat_counts = env_changes.simulate_environment(batch)
        print(f"Batch {batch + 1}:")
        print("Weather:", weather)
        print("Boat Counts:", boat_counts)
        print()