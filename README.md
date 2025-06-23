# 🏥 手术机器人自动导航项目

## 项目概述

本项目旨在开发一个基于AI的手术机器人自动导航系统，实现精准的手术器械控制和路径规划。

## 📋 项目文档

### 核心文档
- [项目技术大纲](docs/overview/手术机器人自动导航项目技术大纲.md) - 整体架构设计
- [文件结构说明](文件结构说明.md) - 项目文件组织说明

### AI组文档
- [第一周任务清单](docs/ai-team/AI组-第一周任务清单.md)
- [医学影像分割技术方案](docs/ai-team/AI组-医学影像分割技术方案.md)

### 控制组文档
- [第一周任务清单](docs/control-team/控制组-第一周任务清单.md)
- [ROS2与CAN集成技术方案](docs/control-team/控制组-ROS2与CAN集成技术方案.md)
- [Day2上午任务完成报告](docs/control-team/Day2上午任务完成报告.md) - ROS2工作空间搭建与基础节点开发

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
```

## 👥 团队分工

### AI组
- 医学影像处理
- 轨迹规划算法
- 深度学习模型开发

### 控制组
- ROS2系统集成
- CAN通信实现
- 机器人运动控制

## 📅 项目进度

- [第一天总结报告](docs/progress/第一天总结报告.md)

## 🛠️ 开发工具

- **IDE**: VSCode/Cursor
- **版本控制**: Git
- **文档**: Markdown
- **通信**: ROS2 + CAN

## 📞 联系方式

- 项目负责人：[待补充]
- AI组负责人：[待补充]
- 控制组负责人：[待补充]

---
*最后更新时间：2024年* 