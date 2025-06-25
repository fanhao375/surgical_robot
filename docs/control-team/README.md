# 🎮 控制组文档索引

## 📚 文档导航

### 📋 任务计划
- [第一周任务清单](./plans/控制组-第一周任务清单.md) - 详细的第一周开发任务
- [第二周任务清单](./plans/控制组-第二周任务清单.md) - 协议系统与硬件集成

### 📊 进度报告  
- [任务报告索引](./reports/README.md) - 详细的每日报告和总结
- [第一周总结](./reports/week1/第一周总结报告.md) - ROS2+CAN系统完成
- [第二周总结](./reports/week2/第二周总结报告.md) - Robot协议实现

### 🔧 技术方案
- [ROS2与CAN集成技术方案](./technical-specs/控制组-ROS2与CAN集成技术方案.md) - 核心架构设计

### 📖 快速参考
- [常用命令参考](./quick-reference/Day2-快速参考.md) - 开发常用命令

## 🎯 当前状态

### ✅ 已完成 (第一周)
- **ROS2系统**: 工作空间、自定义消息、轨迹播放器
- **CAN通信**: 桥接节点、协议栈、仿真测试

### ✅ 已完成 (第二周) 
- **Robot协议**: 自定义协议实现，精度提升100倍
- **可扩展架构**: 支持CANopen和CustomRobot双协议

### 🔄 进行中
- **硬件集成**: 实际CAN设备连接测试
- **AI对接**: 统一控制接口设计

## 🚀 核心技术成果

### 系统架构
```
CSV轨迹 → 轨迹播放器 → ROS2消息 → CAN桥接 → Robot协议 → 电机控制
```

### 关键指标
- **控制精度**: 推进0.01mm，旋转0.1°
- **时序精度**: <10ms轨迹控制
- **通信延迟**: <5ms ROS2↔CAN转换

## 🛠️ 快速启动

### 编译系统
```bash
cd ~/surgical-robot-auto-navigation/surgical_robot_ws
colcon build --packages-select surgical_robot_control
source install/setup.bash
```

### 运行测试
```bash
# 完整系统测试
ros2 launch surgical_robot_control can_test.launch.py

# 轨迹播放测试
ros2 run surgical_robot_control trajectory_player \
  --ros-args -p trajectory_file:=./test_trajectories/test_linear.csv
```

### 协议配置
```bash
# 查看当前协议配置
cat surgical_robot_ws/src/surgical_robot_control/config/custom_robot_protocol_config.yaml

# 切换协议类型 (canopen/custom_robot)
ros2 param set /can_bridge_node protocol_type custom_robot
```

## 📁 关键文件位置

### 协议实现
- `include/surgical_robot_control/custom_robot_protocol.h` - 自定义协议
- `src/updated_protocol_factory.cpp` - 协议工厂
- `config/custom_robot_protocol_config.yaml` - 协议配置

### 核心节点
- `src/trajectory_player.cpp` - 轨迹播放器
- `src/can_bridge_node.cpp` - CAN桥接节点

### 测试文件
- `surgical_robot_ws/test_trajectories/` - 测试轨迹数据

## 🔗 相关文档

- [协议分析文档](../protocols/README.md) - Robot协议详细分析
- [项目文件组织](../文件组织说明.md) - 完整目录结构
- [项目技术大纲](../overview/手术机器人自动导航项目技术大纲.md) - 整体架构

---
*最后更新: 2024年6月25日 - 协议系统完成* 