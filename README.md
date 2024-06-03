# navi_rl

Using reinforcement learning for smart buoys energy consumption.

This study will focus on the energy consumption optimization problem of smart buoys equipped with various sensor devices. By using deep reinforcement learning, it aims to develop a set of energy management strategies that can self-learn and optimize to meet the personalized needs of smart maritime transportation and optimize energy use.

## 项目架构

### 文件和代码块说明：

- **envs/buoy_env.py**
  - 定义了智能浮标的仿真环境，包括状态空间、动作空间和奖励函数。

- **agents/dqn_agent.py**
  - 实现了一个DQN智能体，包括网络定义、学习和决策逻辑。

- **utils/logger.py**
  - 提供日志记录功能，可以在训练和测试时记录实验数据和进度。

- **tests/test_environment.py**
  - 包含对环境模块的单元测试，确保环境行为符合预期。

- **data/training_data.csv**
  - 存放可能用于预训练或分析的数据。

- **models/trained_model.pth**
  - 存储训练后的模型，可用于后续的测试或再训练。

- **checkpoints/checkpoint.pth**
  - 在训练过程中定期保存的模型状态，用于恢复训练或细粒度分析。

- **results/performance.png**
  - 存储生成的性能图表等结果文件，用于报告和分析。

- **scripts/train.py**
  - 启动训练流程的脚本，管理训练的设置、循环和保存逻辑。

- **scripts/evaluate.py**
  - 运行模型评估，生成性能报告和可视化。

- **main.py**
  - 主入口文件，整合上述组件并运行定义好的实验流程。
