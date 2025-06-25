# 协议文档目录

本目录包含手术机器人AI导航系统的协议相关文档。

## 📁 文件组织

### 协议分析文档
- `AI导航协议需求总结.md` - 基于robot协议.xlsx的完整分析
  - AI导航核心协议需求 
  - 可扩展CAN协议架构设计
  - 与AI组的对接接口
  - 实施计划和时间表

### 原始协议文件
- `robot协议.xlsx` - 原始Excel协议文档
- `robot协议_完整版.csv` - 解析后的CSV格式
- `robot协议_AI导航专用.csv` - AI导航相关协议提取

## 🏗️ 协议实现文件位置

### 头文件
```
surgical_robot_ws/src/surgical_robot_control/include/surgical_robot_control/
├── custom_robot_protocol.h          # 自定义robot协议实现
└── can_protocol.h                   # 原有CANopen协议定义
```

### 源文件  
```
surgical_robot_ws/src/surgical_robot_control/src/
├── updated_protocol_factory.cpp     # 更新的协议工厂
├── can_bridge_node.cpp              # CAN桥接节点
└── trajectory_player.cpp            # 轨迹播放器
```

### 配置文件
```
surgical_robot_ws/src/surgical_robot_control/config/
└── custom_robot_protocol_config.yaml # 自定义协议配置
```

## 🎯 协议优先级

基于AI导航需求，协议项目按优先级分为：

### 🔥 Tier 1 - 核心控制 (最高优先级)
- 导丝前进速度控制 (组6序号0) - GW_Go_Data
- 导丝旋转速度控制 (组6序号1)
- 导丝前进位置控制 (组6序号6) 
- 导丝旋转位置控制 (组6序号7)

### ⭐ Tier 2 - 状态反馈 (高优先级)
- 导丝前进位置数据 (组8序号0)
- 导丝旋转位置数据 (组8序号1)

### 🛡️ Tier 3 - 安全监控 (中优先级)
- 急停状态 (组4序号0)
- 导丝当前夹紧力 (组10序号0)
- 导丝阻力传感器 (组10序号1-2)
- M3/M4电机故障码 (组4序号6-7)

### 🤝 Tier 4 - 系统集成 (低优先级)
- 握手 (组3序号2)
- 节点状态控制/反馈 (组5序号0-1)

## 📊 协议精度优势

与标准CANopen协议相比：
- **推进精度**: 1mm → 0.01mm (提升100倍)
- **旋转精度**: 1° → 0.1° (提升10倍)
- **速度精度**: 同样提升100倍和10倍

## 🔗 相关文档

- [控制组第二周任务清单](../control-team/plans/控制组-第二周任务清单.md)
- [ROS2与CAN集成技术方案](../control-team/technical-specs/控制组-ROS2与CAN集成技术方案.md)
- [项目技术大纲](../overview/手术机器人自动导航项目技术大纲.md)

## 🚀 使用说明

1. **开发者**: 参考`AI导航协议需求总结.md`了解完整架构
2. **AI组**: 重点关注Tier 1和Tier 2协议项目
3. **测试**: 使用`custom_robot_protocol_config.yaml`配置测试参数

---
*更新时间: 2024年6月25日* 