# 🧪 测试轨迹文件

## 📁 轨迹文件列表

| 文件名 | 用途 | 持续时间 | 特点 |
|--------|------|----------|------|
| `test_linear.csv` | 线性推送测试 | 2.5秒 | 纯推送运动 (0→5mm) |
| `test_rotation.csv` | 旋转测试 | 2.0秒 | 纯旋转运动 (0→20°) |
| `test_combined.csv` | 复合运动测试 | 4.0秒 | 双轴协调运动 |

## 🔧 使用方法

### 轨迹播放测试
```bash
# 进入ROS2工作空间
cd ~/surgical-robot-auto-navigation/surgical_robot_ws
source install/setup.bash

# 测试线性运动
ros2 run surgical_robot_control trajectory_player \
    --ros-args -p trajectory_file:=$PWD/test_trajectories/test_linear.csv

# 测试旋转运动  
ros2 run surgical_robot_control trajectory_player \
    --ros-args -p trajectory_file:=$PWD/test_trajectories/test_rotation.csv

# 测试复合运动
ros2 run surgical_robot_control trajectory_player \
    --ros-args -p trajectory_file:=$PWD/test_trajectories/test_combined.csv
```

### 监控轨迹播放
```bash
# 查看发布的轨迹命令
ros2 topic echo /trajectory_command

# 查看机器人状态反馈  
ros2 topic echo /robot_state
```

## 📋 CSV文件格式

### 必需列字段
| 列名 | 单位 | 说明 |
|------|------|------|
| `time_ms` | 毫秒 | 时间戳 |
| `push_mm` | 毫米 | 推送位置 |
| `rotate_deg` | 度 | 旋转角度 |
| `velocity_mm_s` | 毫米/秒 | 推送速度 |
| `angular_velocity_deg_s` | 度/秒 | 角速度 |

### 格式要求
- CSV标准格式，UTF-8编码
- 第一行为列标题
- 时间戳严格递增
- 数值字段使用浮点数

## ⚙️ 安全参数

- **最大推送速度**: 2.0mm/s
- **最大角速度**: 10°/s  
- **推送范围**: 0-6mm
- **旋转范围**: 0-20°

## 🔗 相关文档

- [轨迹播放器源码](../src/trajectory_player.cpp)
- [消息定义](../msg/TrajectoryPoint.msg)
- [控制组文档](../../docs/control-team/README.md)

---
*测试轨迹 v1.0 - 支持ROS2轨迹播放器* 