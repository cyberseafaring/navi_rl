# 环境的测试脚本,包含对环境模块的单元测试，确保环境行为符合预期。
import unittest
import numpy as np
from envs.buoy_env import SmartBuoyEnvironment  # 修改导入路径

class TestSmartBuoyEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = SmartBuoyEnvironment()
        self.env.reset()

    def test_initial_state(self):
        state, _ = self.env.reset()
        self.assertEqual(len(state), 7)
        self.assertEqual(state[0], self.env.battery_capacity)  # 初始电池电量应为满电
        self.assertIn(state[1], [0, 1, 2])  # 天气状况应在0, 1, 2之间
        self.assertGreaterEqual(state[2], 0)  # 通信需求应大于等于0
        self.assertEqual(state[3], 1)  # 初始信道质量应为1
        self.assertGreaterEqual(state[4], 0)  # 当前时间应大于等于0
        self.assertLess(state[4], 24)  # 当前时间应小于24
        self.assertGreaterEqual(state[5], 0)  # 当前日期应大于等于0
        self.assertLess(state[5], 7)  # 当前日期应小于7
        self.assertIn(state[6], [0, 1])  # 传感器模式应在0, 1之间

    def test_step_function(self):
        action = self.env.action_space.sample()
        state, reward, done, truncated, info = self.env.step(action)
        self.assertEqual(len(state), 7)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_battery_level(self):
        initial_battery_level = self.env.battery_level
        action = [1, 1, 1, 1, 2]  # 开启所有设备，最高通信QoS
        self.env.step(action)
        self.assertLess(self.env.battery_level, initial_battery_level)  # 电池电量应减少

    def test_solar_generation(self):
        self.env.current_step = 0
        initial_battery_level = self.env.battery_level
        self.env.update_environment()
        self.assertGreaterEqual(self.env.battery_level, initial_battery_level)  # 电池电量应增加或保持不变

    def test_reward_function(self):
        action = [1, 1, 1, 1, 2]  # 开启所有设备，最高通信QoS
        _, reward, _, _, _ = self.env.step(action)
        self.assertIsInstance(reward, float)

    def test_reset_function(self):
        state, _ = self.env.reset()
        self.assertEqual(len(state), 7)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.battery_level, self.env.battery_capacity)

if __name__ == '__main__':
    unittest.main()