# 🏥 手术机器人自动导航项目

## 项目概述

本项目旨在开发一个基于AI的手术机器人自动导航系统，实现精准的手术器械控制和路径规划。

## 📋 项目文档

### 核心文档
- [项目技术大纲](docs/overview/手术机器人自动导航项目技术大纲.md) - 整体架构设计
- [项目进度跟踪](docs/项目进度跟踪.md) - 实时进度监控与里程碑管理
- [文件结构说明](文件结构说明.md) - 项目文件组织说明

### AI组文档
- [第一周任务清单](docs/ai-team/AI组-第一周任务清单.md)
- [医学影像分割技术方案](docs/ai-team/AI组-医学影像分割技术方案.md)

### 控制组文档
- [控制组文档索引](docs/control-team/README.md) - 完整的文档导航
- [第一周任务清单](docs/control-team/plans/控制组-第一周任务清单.md)
- [第一周总结报告](docs/control-team/reports/week1/第一周总结报告.md) - 完整的进度总结
- [ROS2与CAN集成技术方案](docs/control-team/technical-specs/控制组-ROS2与CAN集成技术方案.md)

## 🚀 快速开始

### 环境要求
- Ubuntu 22.04 (WSL2)
- ROS2 Humble
- Python 3.8+
- CUDA 11.8+ (用于AI模型)

### 安装步骤
```bash
# 1. 克隆项目
git clone [项目地址]
cd surgical-robot-auto-navigation

# 2. 整理文件结构（可选）
chmod +x organize_files.sh
./organize_files.sh

# 3. 安装依赖
# AI组依赖
pip install -r requirements-ai.txt

# 控制组依赖
sudo apt install ros-humble-desktop

# 4. 控制组快速测试（确保已安装ROS2）
cd surgical_robot_ws
colcon build --packages-select surgical_robot_control
source install/setup.bash
ros2 launch surgical_robot_control can_test.launch.py
```

## 👥 团队分工

### AI组
- 医学影像处理
- 轨迹规划算法
- 深度学习模型开发

### 控制组 ✅
- **ROS2系统集成** - 工作空间已建立
- **CAN通信实现** - 完整的CANopen协议栈
- **机器人运动控制** - 轨迹播放器和桥接节点完成

## 📅 项目进度

### 第一周成果 (2024.6.22-6.24)
- ✅ **控制组第一周任务** - 100%完成
  - ROS2环境搭建与基础节点开发
  - 自定义消息定义和轨迹格式设计
  - 轨迹播放器实现与测试
  - CAN通信协议栈和桥接节点
- 🔄 **AI组任务** - 进行中

### 关键技术突破
- 建立了完整的 `CSV → ROS2 → CAN → 电机` 数据流管道
- 实现了精确的轨迹时序控制（<10ms精度）
- 完成了CANopen协议的完整封装
- 建立了WSL2环境下的开发调试能力

### 下一阶段计划
- **第二周**: 实际硬件集成与测试
- **第三周**: AI算法与控制系统融合

## 🎯 控制组成果展示

### 可运行的系统组件
```bash
# 1. Hello World节点 - 验证ROS2基础功能
ros2 run surgical_robot_control hello_control

# 2. 轨迹播放器 - 播放CSV轨迹文件
ros2 run surgical_robot_control trajectory_player \
  --ros-args -p trajectory_file:=./test_trajectories/test_linear.csv

# 3. CAN桥接节点 - ROS2到CAN总线的转换
ros2 run surgical_robot_control can_bridge_node \
  --ros-args -p enable_simulation:=true

# 4. 完整系统测试 - 轨迹播放+CAN通信
ros2 launch surgical_robot_control can_test.launch.py

# 5. CAN消息测试 - 验证协议封装
ros2 run surgical_robot_control can_message_test
```

### 技术架构
```
📂 surgical_robot_ws/
├── 🎯 轨迹播放器 (trajectory_player)
├── 🔄 CAN桥接节点 (can_bridge_node)  
├── 📡 自定义消息 (TrajectoryPoint/RobotState)
├── ⚙️ CANopen协议栈 (can_protocol.h)
└── 🧪 测试框架 (完整的单元与集成测试)
```

## 🛠️ 开发工具

- **IDE**: VSCode/Cursor
- **版本控制**: Git
- **构建系统**: CMake + Colcon
- **文档**: Markdown
- **通信协议**: ROS2 + CANopen

## 📞 联系方式

- 项目负责人：[待补充]
- AI组负责人：[待补充]
- 控制组负责人：[待补充]

---
*最后更新时间：2024年6月24日 - 控制组第一周任务完成* 