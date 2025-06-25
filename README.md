# 🏥 手术机器人自动导航项目

## 🎯 项目概述

基于DSA医学影像+AI算法实现介入手术机器人自动导航，完成导丝推送和旋转动作自动化。

### 核心技术链路
```
DSA影像 → 血管分割 → 路径规划 → 运动控制 → 机器人执行
```

## 📚 文档导航

| 模块 | 文档 | 说明 |
|------|------|------|
| **项目概览** | [技术大纲](docs/overview/手术机器人自动导航项目技术大纲.md) | 整体架构设计 |
| **协议文档** | [协议文档](docs/protocols/README.md) | Robot协议分析和实现 |
| **控制组** | [控制组文档](docs/control-team/README.md) | ROS2+CAN系统开发 |
| **AI组** | [AI组文档](docs/ai-team/) | 医学影像处理和轨迹规划 |
| **项目管理** | [文件组织说明](docs/文件组织说明.md) | 完整项目结构 |

## 🚀 快速开始

### 环境要求
- **操作系统**: Ubuntu 22.04 (WSL2支持)
- **ROS版本**: ROS2 Humble  
- **Python**: 3.8+
- **CUDA**: 11.8+ (AI模型训练)

### 控制组快速启动
```bash
# 1. 进入ROS2工作空间
cd surgical_robot_ws

# 2. 编译控制包
colcon build --packages-select surgical_robot_control
source install/setup.bash

# 3. 启动测试系统
ros2 launch surgical_robot_control can_test.launch.py
```

### AI组快速启动
```bash
# 1. 安装AI依赖
pip install -r requirements-ai.txt

# 2. 启动医学影像处理
python src/ai/vessel_segmentation.py

# 3. 启动轨迹规划
python src/ai/trajectory_planning.py
```

## 🏗️ 系统架构

### 控制系统架构
```
CSV轨迹 → 轨迹播放器 → ROS2消息 → CAN桥接 → Robot协议 → 电机控制
```

### AI系统架构  
```
DSA影像 → 血管分割 → 3D重建 → 路径规划 → 轨迹生成 → CSV输出
```

## 📈 项目进度

### 🎯 当前状态 (2024年6月)
- ✅ **控制组**: ROS2+CAN系统完成，Robot协议实现
- 🔄 **AI组**: 医学影像分割算法开发中
- 🔄 **系统集成**: 准备AI+控制系统联调

### 🚀 核心技术突破
- **控制精度**: 推进0.01mm，旋转0.1° (比标准协议精度提升100倍)
- **通信链路**: 完整的ROS2↔CAN↔Robot协议栈
- **轨迹控制**: <10ms时序精度的轨迹播放系统

## 👥 团队分工

| 团队 | 主要职责 | 关键交付 |
|------|---------|----------|
| **AI组** | 医学影像处理、轨迹规划 | 血管分割模型、路径规划算法 |
| **控制组** | 机器人运动控制、系统集成 | ROS2控制系统、CAN通信栈 |

## 🛠️ 开发环境

- **开发工具**: VSCode/Cursor + ROS2扩展
- **版本控制**: Git
- **构建工具**: CMake + Colcon (ROS2)
- **调试工具**: RQt + RViz2
- **协议工具**: CANopen分析仪

## 🔗 相关链接

- **协议分析**: 查看 `docs/protocols/AI导航协议需求总结.md`
- **技术实现**: 参考 `surgical_robot_ws/src/surgical_robot_control/`
- **开发文档**: 浏览 `docs/control-team/` 和 `docs/ai-team/`

---
*最后更新: 2024年6月25日 - 协议系统完成* 