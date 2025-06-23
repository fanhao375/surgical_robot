# 测试轨迹文件说明

## 📁 文件列表

### 基础轨迹文件

1. **`test_linear.csv`** - 线性推送运动
   - 总时长：2.5秒
   - 推送距离：0 → 5.0mm
   - 旋转角度：保持0°
   - 推送速度：恒定2.0mm/s
   - 用途：测试推送轴单独运动

2. **`test_rotation.csv`** - 纯旋转运动
   - 总时长：2.0秒
   - 推送距离：保持0mm
   - 旋转角度：0° → 20°
   - 角速度：恒定10°/s
   - 用途：测试旋转轴单独运动

3. **`test_combined.csv`** - 复合运动
   - 总时长：4.0秒
   - 推送距离：0 → 6.0mm
   - 旋转角度：0° → 20°
   - 推送速度：恒定1.5mm/s
   - 角速度：恒定5°/s
   - 用途：测试双轴协调运动

## 📋 CSV文件格式说明

### 列定义

| 列名 | 单位 | 说明 |
|------|------|------|
| `time_ms` | 毫秒 | 时间戳，从0开始递增 |
| `push_mm` | 毫米 | 推送位置（绝对位置） |
| `rotate_deg` | 度 | 旋转角度（绝对角度） |
| `velocity_mm_s` | 毫米/秒 | 推送速度 |
| `angular_velocity_deg_s` | 度/秒 | 角速度 |

### 格式要求

1. **CSV标准格式**：逗号分隔，第一行为列标题
2. **时间递增**：时间戳必须严格递增
3. **数值格式**：所有数值字段使用浮点数
4. **编码格式**：UTF-8编码

## 🔗 与TrajectoryPoint消息的映射

CSV文件字段与ROS2消息字段的对应关系：

```
CSV字段               → TrajectoryPoint消息字段
time_ms              → timestamp (转换为秒)
push_mm              → push_position
rotate_deg           → rotate_angle  
velocity_mm_s        → push_velocity
angular_velocity_deg_s → angular_velocity
```

## 🛠️ 验证工具

使用 `validate_trajectories.py` 脚本验证轨迹文件：

```bash
python3 validate_trajectories.py
```

验证内容包括：
- CSV格式正确性
- 必需列完整性
- 数据类型有效性
- 时间戳递增性
- 数值合理性检查

## 🎯 使用方法

### Day4轨迹播放测试

这些文件将在Day4用于测试轨迹播放器：

```bash
# 测试线性运动
ros2 run surgical_robot_control trajectory_player \
    --ros-args -p trajectory_file:=$PWD/test_linear.csv

# 测试旋转运动  
ros2 run surgical_robot_control trajectory_player \
    --ros-args -p trajectory_file:=$PWD/test_rotation.csv

# 测试复合运动
ros2 run surgical_robot_control trajectory_player \
    --ros-args -p trajectory_file:=$PWD/test_combined.csv
```

### 监控轨迹播放

在另一个终端监控发布的轨迹点：

```bash
ros2 topic echo /trajectory_command
```

## 📐 设计考量

### 安全参数

- **最大推送速度**：2.0mm/s（适合手术环境）
- **最大角速度**：10°/s（避免过快旋转）
- **推送范围**：0-6mm（典型手术推送距离）
- **旋转范围**：0-20°（有限角度避免缠绕）

### 时间精度

- **采样间隔**：500ms（适合测试，实际可更密集）
- **时间戳精度**：毫秒级（满足控制要求）

### 扩展性

轨迹文件格式支持：
- 添加更多运动轴
- 增加力反馈参数
- 集成安全限制参数

## 🔄 版本历史

- v1.0 (Day3下午) - 初始版本，包含基础三种轨迹类型
- 支持后续扩展复杂轨迹模式

---

**创建时间**: Day3下午  
**验证状态**: ✅ 所有文件通过验证  
**准备状态**: ✅ Day4轨迹播放器就绪 