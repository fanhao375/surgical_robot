# 手术机器人通信协议完整规范 V2.0

## 协议概述
本文档是手术机器人系统的完整通信协议规范，基于CAN总线通信，包含11个功能组共91个控制项目的详细定义。

### 协议特点
- **高精度控制**：推进精度0.01mm，旋转精度0.1°
- **安全优先**：多重急停机制和传感器监控
- **模块化设计**：11个功能组覆盖系统全部功能
- **实时反馈**：位置、力、状态全方位反馈

## 完整协议内容

### 组1: 空余项

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | Globle类型代号 | Node.Type | 只读 | int | 0x8001 |
| 1 | 对象占用空间 | Node.Size | 只读 | u32 | 对象占用空间大小 |
| 2 | 无 | - | - | - | - |

### 组2: 版本控制

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 对象版本号 | Node.Version | 只读 | u32 | 对象版本号: V:X,Y,Z,B /  |
| 1 | 节点固件升级控制 | - | 读写 | int32 | 0、无 / 1、节点软件复位，准备接受APP / 2、运行APP程序 |
| 2 | 固件更新状态 | - | 读写 | int32 | 0、无； / 1、上电且在运行bootloder / 2、APP接收中 / 3、APP接收完成 / ... |
| 3 | 系统状态 | - | 只读 | int32 | 0.无 / 1.开机中 / 10.用户登录中 / 20 病例输入 / 30系统初始化3 / 40消毒... |

### 组3: 开关机、握手状态

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 开关机状态 | PowerState | 读写 | int32 | 0、开机等待； / 1、开机完成； / 2、关机完成 |
| 1 | 一键开关机指令 | - | 读写 | int32 | 0.待机 / 1.系统开机 / 2.正常关机 / 3.强制关机 |
| 2 | 握手 | Handshake | 读写 | int32 | 0~255递增 |
| 3 | 无 | - | - | int32 | - |

### 组4: 急停和错误

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 急停状态 | Emergency | 读写 | int32 | 0、非急停状态； / 1、下位机广播急停 |
| 1 | 节点状态 | Node.Status | 读写 | int32 | 0、正常状态； / 1、错误状态 |
| 2 | 错误代码 | Node.ErNo | 读写 | int32 | 错误代码号 |
| 3 | 错误清除 | Node.ErClr | 读写 | int32 | 0、无 / 1、错误清除 |
| 4 | M1_导丝夹紧电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 5 | M2_导丝夹紧电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 6 | M3_导丝推进电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 7 | M4_导丝旋转电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 8 | M5_导丝旋转电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 9 | M6_支架推进电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 10 | M7_辅助轮压紧电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 11 | M8_辅助轮旋转电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 12 | M9_辅助轮压紧电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 13 | M10_辅助轮旋转电机故障码 | - | 只读 | uint32 | 值参考：电机故障码 |
| 14 | 无 | - | - | - | - |

### 组5: 节点状态控制

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 子节点状态控制 /  | BoxStateCtrl | 读写 | int32 | 0、消毒盒复位； / 1、消毒盒初始化 / 2、导丝、导管 夹紧 / 3、导丝、导管 松开 / 4、... |
| 1 | 子节点状态反馈 | ImpactState | 读写 | int32 | 0、消毒盒复位完成； / 1、消毒盒初始化完成 / 2、导丝、导管 夹紧完成 / 3、导丝、导管 松... |
| 2 | 系统状态 | 用于系统状态 | 读写 | int32 | 用于屏幕状态显示 |
| 3 | 电磁铁状态 | Electromagnet | 读写 | int32 | 0、无 / 1、电磁铁上电 / 2、电磁铁失电 |
| 4 | 电机供电控制 | - | 读写 | int32 | 0、电机电源关闭 / 1、电机电源打开 |
| 5 | 氛围灯颜色 | - | 读写 | int32 | 0、熄灭 / 1、绿色 / 2、蓝色 / 3、黄色 / 4、红色 |
| 6 | 脚闸（本地端）状态（透视） | - | 只写 | int32 | 0、弹起状态 / 1、压下状态 / 2、错误状态 |
| 7 | 脚闸（本地端）状态（曝光） | - | 只写 | int32 | 0、弹起状态 / 1、压下状态 / 2、错误状态 |
| 8 | 旋转辅助轮控制 | - | 读写 | int32 | 0、无 / 1、无 / 2、辅助轮压紧 / 3、辅助轮松开 |
| 9 | 旋转辅助轮状态反馈 | - | 读写 | int32 | 0、无 / 1、无 / 2、辅助轮压紧完成 / 3、辅助轮松开完成 |
| 10 | 刹车控制 | - | 读写 | int32 | 0、无 / 1、刹车 / 2、解锁 |
| 11 | 支架夹紧松开控制 | - | 读写 | int32 | 0、1、无 / 2、支架夹紧 / 3、支架松开 |
| 12 | 支架夹紧松开状态反馈 | - | 读写 | int32 | 0、1、无 / 2、支架压紧完成 / 3、支架松开完成 |
| 13 | 控制盒音量控制 | - | 只写 | - | 0、无声音 / 1~5挡位 |

### 组6: 运动控制 ⭐ (AI导航核心)

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 导丝前进速度控制 | GW_Go_Data | 读写 | int32 | data |
| 1 | 导丝旋转速度控制 | - | 读写 | int32 | data |
| 2 | 造影导管前进速度控制 | - | 读写 | int32 | data |
| 3 | 造影导管旋转速度控制 | - | 读写 | int32 | data |
| 4 | 机械臂整体运动速度控制 | - | 读写 | int32 | data |
| 5 | 支架导管前进速度控制 | - | 读写 | int32 | data |
| 6 | 导丝前进位置控制 | - | 读写 | int32 | data |
| 7 | 导丝旋转位置控制 | - | 读写 | int32 | data |
| 8 | 造影导管前进位置控制 | - | 读写 | int32 | data |
| 9 | 造影导管旋转位置控制 | - | 读写 | int32 | data |
| 10 | 机械臂整体运动位置控制 | - | 读写 | int32 | data |
| 11 | 支架导管前进位置控制 | - | 读写 | int32 | data |
| 12 | 无 | - | - | - | - |

#### 重要说明
- **速度控制**：单位为0.01mm/s（推进）和0.1°/s（旋转）
- **位置控制**：单位为0.01mm（推进）和0.1°（旋转）
- **方向定义**：推进（-左，+右），旋转（-顺时针，+逆时针）

### 组7: 摇杆滚轮输入

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 摇杆1数据 | JoyStick[0] | 读写 | int32 | 数据范围：[-15,+15] |
| 1 | 摇杆2数据 | JoyStick[1] | 读写 | int32 | 数据范围：[-15,+15] |
| 2 | 摇杆3数据 | JoyStick[2] | 读写 | int32 | 数据范围：[-15,+15] |
| 3 | 滚轮1数据 | Encoder[0] | 读写 | int32 | data |
| 4 | 滚轮2数据 | - | 读写 | int32 | data |
| 5 | 无 | - | - | int32 | - |

### 组8: 运动信息 ⭐ (AI导航反馈)

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 导丝前进位置数据 | - | 只读 | int32 | data |
| 1 | 导丝旋转位置数据 | - | 只读 | int32 | data |
| 2 | 造影导管前进电机位置数据 | - | 只读 | int32 | data |
| 3 | 造影导管旋转电机位置数据 | - | 只读 | int32 | data |
| 4 | 机械臂整体前进电机位置数据 | - | 只读 | int32 | data |
| 5 | 支架导管前进电机位置数据 | - | 只读 | int32 | data |
| 6 | 无 | - | - | int32 | - |

### 组9: 开关量输入等信息

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 消毒盒上盖状态 | - | 读写 | int32 | 0、初始状态； / 1、消毒盒被打开 / 2、消毒盒关闭 |
| 1 | 大臂立柱限位 | - | 读写 | int32 | 0、无； / 1、立柱限位左触发 / 2、立柱限位右触发 / 3、立柱限位异常 |
| 2 | 光电开关状态 | - | 只读 | int32 | 2进制表示光电状态 |
| 3 | 脚闸（远程端）状态（透视） | - | 只读 | int32 | 0、弹起状态 / 1、压下状态 / 2、错误状态 |
| 4 | 脚闸（远程端）状态（曝光） | - | 只读 | int32 | 0、弹起状态 / 1、压下状态 / 2、错误状态 |
| 5 | 刹车解除按键状态 | - | 只读 | int32 | 0、无 / 1、刹车 / 2、解锁 |
| 6 | 大臂前进后退按键状态 | - | 只读 | int32 | 0、无输入 / 1、导丝机构前进 / 2、导丝机构后退 |
| 7 | 换盒键状态 | - | 只读 | int32 | 0、无 / 1、按键触发 |
| 8 | 导丝夹紧/松开按键 | - | 只读 | int32 | 0、无 / 1、导丝夹紧 / 2、导丝松开 |
| 9 | 支架夹紧/松开按键 | - | 只读 | int32 | 0、无 / 1、支架夹紧 / 2、支架松开 |
| 10 | 消毒盒安装状态 | - | 只读 | int32 | 0、无消毒盒安装 / 1、消毒盒已安装 |
| 11 | 串口屏按键禁用 | - | 读写 | int32 | 0、初始状态全禁用 / 详见串口屏按键禁用 |
| 12 | 无 | - | - | - | - |

### 组10: 传感器数据 ⭐ (AI导航安全)

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 导丝当前夹紧力 | Force[0] | 只读 | int32 | data |
| 1 | 导丝阻力主左 / （左边主动端传感器） | - | 只读 | int32 | data |
| 2 | 导丝阻力主右 / （右边主动端传感器） | - | 只读 | int32 | data |
| 3 | 导丝阻力传感器被左 / （左边被动端传感器） | - | 只读 | int32 | data |
| 4 | 导丝阻力传感器被右 / （右边被动端传感器） | - | 只读 | int32 | data  |
| 5 | 造影导管压紧力传感器 | - | 只读 | int32 | data |
| 6 | 导丝前进编码器角度值 / （暂定：左边加紧，数值增加，右边反之） /  | - | 只读 | int32 | data |
| 7 | 编码器2角度值 | - | 只读 | int32 | data |
| 8 | 编码器3角度值 | - | 只读 | int32 | data |

#### 传感器说明
- **力传感器**：单位为g，需转换为N（牛顿）
- **阻力传感器**：分主动端和被动端，用于监控导丝运动阻力
- **编码器**：提供精确的位置反馈

### 组11: 配置信息

| 序号 | 信息项 | 信息项代号 | 操作类型 | 数据类型 | 信息内容 |
|-----|-------|-----------|---------|---------|----------|
| 0 | 导丝夹紧力配置 | - | 读写 | int32 | data |
| 1 | 辅助轮压紧力配置 | - | 读写 | int32 | data |
| 2 | Can阻抗适配 | - | 读写 | int32 | 0、小臂已连接 / 1、小臂未连接 |

## 协议统计

- **总组数**: 11
- **总项目数**: 91

### 各组项目统计

| 组号 | 组名 | 项目数 | 主要功能 |
|-----|------|-------|----------|
| 1 | 空余项 | 3 | 系统基础信息 |
| 2 | 版本控制 | 4 | 版本管理和固件升级 |
| 3 | 开关机、握手状态 | 4 | 系统启动和通信握手 |
| 4 | 急停和错误 | 15 | 急停和错误处理 |
| 5 | 节点状态控制 | 14 | 电机和系统状态控制 |
| 6 | 运动控制 | 13 | 精确运动控制（速度/位置） |
| 7 | 摇杆滚轮输入 | 6 | 手动操作输入 |
| 8 | 运动信息 | 7 | 实时位置反馈 |
| 9 | 开关量输入等信息 | 13 | 开关和按键状态 |
| 10 | 传感器数据 | 9 | 力和位置传感器 |
| 11 | 配置信息 | 3 | 系统配置参数 |

## AI导航系统关键接口

### 核心控制指令（组6）
```python
# 导丝推进控制
push_speed = target_speed_mm_s * 100  # 转换为0.01mm/s
push_position = target_pos_mm * 100   # 转换为0.01mm

# 导丝旋转控制
rotate_speed = target_speed_deg_s * 10  # 转换为0.1°/s
rotate_position = target_angle_deg * 10  # 转换为0.1°
```

### 关键反馈数据（组8）
```python
# 位置反馈
actual_push_mm = push_position_data / 100    # 从0.01mm转换
actual_angle_deg = rotate_position_data / 10  # 从0.1°转换
```

### 安全监控（组4、组10）
```python
# 急停检测
if emergency_status == 1:
    stop_all_motors()

# 阻力监控
if force_sensor > threshold:
    reduce_speed_or_stop()
```

---
*本文档基于robot协议.xlsx完整解析生成，包含所有91个控制项目的详细信息*
*生成时间：2024年*
