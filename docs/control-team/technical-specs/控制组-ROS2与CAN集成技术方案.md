# 控制组技术方案 - ROS2与CAN集成

## 一、任务概述

### 1.1 目标

实现从轨迹文件到机器人运动的完整控制链路，为AI组提供标准化的控制接口。

### 1.2 输入输出定义

- **输入**：轨迹文件（CSV/JSON格式），包含时间序列的推送距离和旋转角度
- **输出**：CAN指令控制机器人完成相应动作

### 1.3 核心任务

1. 搭建ROS2开发环境
2. 实现轨迹解析与播放
3. 开发CAN通信桥接节点
4. 完成运动控制与安全监控

## 二、开发环境搭建（Day 1-2）

### 2.1 WSL2环境配置

```bash
# 1. 安装WSL2 Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# 2. 更新系统
sudo apt update && sudo apt upgrade -y

# 3. 安装基础开发工具
sudo apt install -y build-essential cmake git python3-pip
```

### 2.2 ROS2 Humble安装

```bash
# 1. 设置源
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y

# 2. 添加ROS2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 3. 添加仓库
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 4. 安装ROS2
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-ros2-control

# 5. 环境配置
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2.3 CAN工具安装

```bash
# SocketCAN工具
sudo apt install -y can-utils

# PCAN驱动（如果使用PCAN硬件）
# 下载对应驱动：https://www.peak-system.com/fileadmin/media/linux/
```

## 三、项目结构设计（Day 3）

### 3.1 工作空间创建

```bash
# 创建工作空间
mkdir -p ~/surgical_robot_ws/src
cd ~/surgical_robot_ws

# 创建功能包
cd src
ros2 pkg create --build-type ament_cmake surgical_robot_control \
  --dependencies rclcpp std_msgs geometry_msgs sensor_msgs

# 项目结构
surgical_robot_control/
├── include/surgical_robot_control/
│   ├── trajectory_player.hpp
│   ├── can_bridge.hpp
│   └── safety_monitor.hpp
├── src/
│   ├── trajectory_player.cpp
│   ├── can_bridge_node.cpp
│   ├── safety_monitor.cpp
│   └── main.cpp
├── config/
│   ├── robot_config.yaml
│   └── can_config.yaml
├── launch/
│   └── robot_control.launch.py
├── msg/
│   ├── TrajectoryPoint.msg
│   └── RobotState.msg
└── test/
    └── test_trajectories/
```

## 四、轨迹数据接口定义（Day 4）

### 4.1 轨迹文件格式（CSV）

```csv
# trajectory.csv
# time_ms, push_mm, rotate_deg, velocity_mm_s, angular_velocity_deg_s
0,0.0,0.0,5.0,10.0
100,0.5,2.0,5.0,10.0
200,1.0,4.0,5.0,10.0
300,1.5,6.0,5.0,10.0
```

### 4.2 ROS2消息定义

```bash
# msg/TrajectoryPoint.msg
float64 timestamp
float64 push_position      # mm
float64 rotate_angle       # degrees
float64 push_velocity      # mm/s
float64 angular_velocity   # deg/s

# msg/RobotState.msg
float64 timestamp
float64 actual_push_position
float64 actual_rotate_angle
float64 push_force
float64 rotate_torque
uint8 status              # 0:idle, 1:moving, 2:error
string error_message
```

## 五、核心节点实现（Day 5-8）

### 5.1 轨迹播放节点

```cpp
// trajectory_player.cpp
#include "rclcpp/rclcpp.hpp"
#include "surgical_robot_control/msg/trajectory_point.hpp"
#include <fstream>
#include <sstream>
#include <vector>

class TrajectoryPlayer : public rclcpp::Node
{
public:
    TrajectoryPlayer() : Node("trajectory_player")
    {
        // 发布器
        trajectory_pub_ = this->create_publisher<TrajectoryPoint>(
            "trajectory_command", 10);
      
        // 定时器 - 100Hz
        timer_ = this->create_wall_timer(
            10ms, std::bind(&TrajectoryPlayer::timer_callback, this));
      
        // 参数
        this->declare_parameter("trajectory_file", "");
        load_trajectory();
    }

private:
    void load_trajectory()
    {
        std::string filename = this->get_parameter("trajectory_file").as_string();
        std::ifstream file(filename);
        std::string line;
      
        // 跳过标题行
        std::getline(file, line);
      
        while (std::getline(file, line))
        {
            TrajectoryPoint point;
            std::stringstream ss(line);
            char comma;
          
            ss >> point.timestamp >> comma
               >> point.push_position >> comma
               >> point.rotate_angle >> comma
               >> point.push_velocity >> comma
               >> point.angular_velocity;
          
            trajectory_.push_back(point);
        }
      
        RCLCPP_INFO(this->get_logger(), 
            "Loaded %zu trajectory points", trajectory_.size());
    }
  
    void timer_callback()
    {
        if (current_index_ < trajectory_.size())
        {
            auto point = trajectory_[current_index_];
            point.timestamp = this->now().seconds();
            trajectory_pub_->publish(point);
          
            // 检查是否到达下一个点的时间
            if (current_index_ + 1 < trajectory_.size())
            {
                auto next_time = trajectory_[current_index_ + 1].timestamp;
                auto current_time = trajectory_[current_index_].timestamp;
              
                if (elapsed_time_ >= (next_time - current_time))
                {
                    current_index_++;
                    elapsed_time_ = 0;
                }
            }
          
            elapsed_time_ += 0.01; // 10ms
        }
    }
  
    rclcpp::Publisher<TrajectoryPoint>::SharedPtr trajectory_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<TrajectoryPoint> trajectory_;
    size_t current_index_ = 0;
    double elapsed_time_ = 0;
};
```

### 5.2 CAN桥接节点

```cpp
// can_bridge_node.cpp
#include "rclcpp/rclcpp.hpp"
#include "surgical_robot_control/msg/trajectory_point.hpp"
#include "surgical_robot_control/msg/robot_state.hpp"
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <cstring>

class CANBridge : public rclcpp::Node
{
public:
    CANBridge() : Node("can_bridge")
    {
        // 订阅轨迹命令
        trajectory_sub_ = this->create_subscription<TrajectoryPoint>(
            "trajectory_command", 10,
            std::bind(&CANBridge::trajectory_callback, this, std::placeholders::_1));
      
        // 发布机器人状态
        state_pub_ = this->create_publisher<RobotState>("robot_state", 10);
      
        // 初始化CAN
        init_can();
      
        // 状态反馈定时器 - 50Hz
        feedback_timer_ = this->create_wall_timer(
            20ms, std::bind(&CANBridge::read_can_feedback, this));
    }

private:
    void init_can()
    {
        // 创建socket
        can_socket_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (can_socket_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create CAN socket");
            return;
        }
      
        // 绑定到can0接口
        struct ifreq ifr;
        strcpy(ifr.ifr_name, "can0");
        ioctl(can_socket_, SIOCGIFINDEX, &ifr);
      
        struct sockaddr_can addr;
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
      
        if (bind(can_socket_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to bind CAN socket");
            return;
        }
      
        RCLCPP_INFO(this->get_logger(), "CAN interface initialized");
    }
  
    void trajectory_callback(const TrajectoryPoint::SharedPtr msg)
    {
        // 转换为CAN消息
        struct can_frame frame;
      
        // 推送电机命令 (CAN ID: 0x601)
        frame.can_id = 0x601;
        frame.can_dlc = 8;
      
        // 位置命令 (SDO写入0x6081)
        int32_t push_counts = mm_to_counts(msg->push_position);
        frame.data[0] = 0x23;  // SDO下载命令
        frame.data[1] = 0x81;  // 索引低字节
        frame.data[2] = 0x60;  // 索引高字节
        frame.data[3] = 0x00;  // 子索引
        memcpy(&frame.data[4], &push_counts, 4);
      
        send_can_frame(frame);
      
        // 旋转电机命令 (CAN ID: 0x602)
        frame.can_id = 0x602;
        int32_t rotate_counts = deg_to_counts(msg->rotate_angle);
        memcpy(&frame.data[4], &rotate_counts, 4);
      
        send_can_frame(frame);
    }
  
    void send_can_frame(const struct can_frame& frame)
    {
        if (write(can_socket_, &frame, sizeof(frame)) != sizeof(frame)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to send CAN frame");
        }
    }
  
    void read_can_feedback()
    {
        struct can_frame frame;
        int nbytes = read(can_socket_, &frame, sizeof(frame));
      
        if (nbytes > 0) {
            // 解析反馈数据
            RobotState state;
            state.timestamp = this->now().seconds();
          
            // 根据CAN ID判断数据类型
            if (frame.can_id == 0x181) {  // 推送电机反馈
                int32_t position;
                memcpy(&position, &frame.data[0], 4);
                state.actual_push_position = counts_to_mm(position);
            }
            else if (frame.can_id == 0x182) {  // 旋转电机反馈
                int32_t angle;
                memcpy(&angle, &frame.data[0], 4);
                state.actual_rotate_angle = counts_to_deg(angle);
            }
          
            state_pub_->publish(state);
        }
    }
  
    // 单位转换函数
    int32_t mm_to_counts(double mm) { 
        return static_cast<int32_t>(mm * 1000.0); // 1mm = 1000 counts
    }
  
    double counts_to_mm(int32_t counts) { 
        return counts / 1000.0; 
    }
  
    int32_t deg_to_counts(double deg) { 
        return static_cast<int32_t>(deg * 100.0); // 1deg = 100 counts
    }
  
    double counts_to_deg(int32_t counts) { 
        return counts / 100.0; 
    }
  
    int can_socket_;
    rclcpp::Subscription<TrajectoryPoint>::SharedPtr trajectory_sub_;
    rclcpp::Publisher<RobotState>::SharedPtr state_pub_;
    rclcpp::TimerBase::SharedPtr feedback_timer_;
};
```

## 六、安全监控实现（Day 9）

### 6.1 安全监控节点

```cpp
// safety_monitor.cpp
class SafetyMonitor : public rclcpp::Node
{
public:
    SafetyMonitor() : Node("safety_monitor")
    {
        // 配置参数
        this->declare_parameter("max_push_velocity", 10.0);     // mm/s
        this->declare_parameter("max_angular_velocity", 30.0);  // deg/s
        this->declare_parameter("max_push_force", 5.0);         // N
        this->declare_parameter("position_error_threshold", 2.0); // mm
      
        // 订阅
        trajectory_sub_ = this->create_subscription<TrajectoryPoint>(
            "trajectory_command", 10,
            std::bind(&SafetyMonitor::check_trajectory, this, std::placeholders::_1));
          
        state_sub_ = this->create_subscription<RobotState>(
            "robot_state", 10,
            std::bind(&SafetyMonitor::check_state, this, std::placeholders::_1));
      
        // 发布安全命令
        safety_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "emergency_stop", 10);
    }
  
private:
    void check_trajectory(const TrajectoryPoint::SharedPtr msg)
    {
        double max_push_vel = this->get_parameter("max_push_velocity").as_double();
        double max_angular_vel = this->get_parameter("max_angular_velocity").as_double();
      
        if (msg->push_velocity > max_push_vel || 
            msg->angular_velocity > max_angular_vel) {
            trigger_emergency_stop("Velocity limit exceeded");
        }
    }
  
    void check_state(const RobotState::SharedPtr msg)
    {
        // 检查跟踪误差
        if (last_command_) {
            double push_error = std::abs(msg->actual_push_position - 
                                       last_command_->push_position);
            double threshold = this->get_parameter("position_error_threshold").as_double();
          
            if (push_error > threshold) {
                trigger_emergency_stop("Position tracking error too large");
            }
        }
      
        // 检查力/力矩
        double max_force = this->get_parameter("max_push_force").as_double();
        if (msg->push_force > max_force) {
            trigger_emergency_stop("Force limit exceeded");
        }
    }
  
    void trigger_emergency_stop(const std::string& reason)
    {
        RCLCPP_ERROR(this->get_logger(), "EMERGENCY STOP: %s", reason.c_str());
      
        auto stop_msg = std_msgs::msg::Bool();
        stop_msg.data = true;
        safety_pub_->publish(stop_msg);
    }
};
```

## 七、配置文件（Day 10）

### 7.1 机器人配置

```yaml
# config/robot_config.yaml
robot:
  # 机械参数
  push_axis:
    counts_per_mm: 1000
    max_position: 300.0  # mm
    min_position: 0.0
    max_velocity: 10.0   # mm/s
    max_acceleration: 50.0  # mm/s^2
  
  rotate_axis:
    counts_per_degree: 100
    max_angle: 180.0     # degrees
    min_angle: -180.0
    max_velocity: 30.0   # deg/s
    max_acceleration: 100.0  # deg/s^2
  
  # 安全参数
  safety:
    max_force: 5.0       # N
    max_torque: 0.5      # Nm
    position_error_threshold: 2.0  # mm
    angle_error_threshold: 5.0     # degrees
    watchdog_timeout: 100  # ms
```

### 7.2 CAN配置

```yaml
# config/can_config.yaml
can:
  interface: "can0"
  bitrate: 1000000  # 1Mbps
  
  # CANopen节点配置
  nodes:
    push_motor:
      node_id: 1
      heartbeat_time: 100  # ms
      pdo_mapping:
        rpdo1: [0x6040, 0x6081]  # 控制字, 目标位置
        tpdo1: [0x6041, 0x6064]  # 状态字, 实际位置
      
    rotate_motor:
      node_id: 2
      heartbeat_time: 100
      pdo_mapping:
        rpdo1: [0x6040, 0x6081]
        tpdo1: [0x6041, 0x6064]
```

## 八、启动文件（Day 11）

```python
# launch/robot_control.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # 获取配置文件路径
    config_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        '..', 'config')
  
    return LaunchDescription([
        # 声明参数
        DeclareLaunchArgument(
            'trajectory_file',
            default_value='',
            description='Path to trajectory CSV file'
        ),
      
        # 轨迹播放节点
        Node(
            package='surgical_robot_control',
            executable='trajectory_player',
            name='trajectory_player',
            parameters=[{
                'trajectory_file': LaunchConfiguration('trajectory_file')
            }],
            output='screen'
        ),
      
        # CAN桥接节点
        Node(
            package='surgical_robot_control',
            executable='can_bridge',
            name='can_bridge',
            parameters=[
                os.path.join(config_dir, 'robot_config.yaml'),
                os.path.join(config_dir, 'can_config.yaml')
            ],
            output='screen'
        ),
      
        # 安全监控节点
        Node(
            package='surgical_robot_control',
            executable='safety_monitor',
            name='safety_monitor',
            parameters=[
                os.path.join(config_dir, 'robot_config.yaml')
            ],
            output='screen'
        ),
    ])
```

## 九、测试方案（Day 12-14）

### 9.1 单元测试

```bash
# 1. CAN通信测试
candump can0  # 监控CAN总线
cansend can0 601#2340060000000000  # 发送测试帧

# 2. 节点通信测试
ros2 topic list
ros2 topic echo /trajectory_command
ros2 topic echo /robot_state

# 3. 轨迹回放测试
ros2 launch surgical_robot_control robot_control.launch.py \
    trajectory_file:=test_linear.csv
```

### 9.2 集成测试用例

#### 测试轨迹1：直线推进

```csv
# test_linear.csv
time_ms,push_mm,rotate_deg,velocity_mm_s,angular_velocity_deg_s
0,0.0,0.0,5.0,0.0
1000,5.0,0.0,5.0,0.0
2000,10.0,0.0,5.0,0.0
3000,15.0,0.0,5.0,0.0
4000,20.0,0.0,5.0,0.0
```

#### 测试轨迹2：旋转测试

```csv
# test_rotation.csv
time_ms,push_mm,rotate_deg,velocity_mm_s,angular_velocity_deg_s
0,0.0,0.0,0.0,10.0
1000,0.0,10.0,0.0,10.0
2000,0.0,20.0,0.0,10.0
3000,0.0,30.0,0.0,10.0
4000,0.0,40.0,0.0,10.0
```

#### 测试轨迹3：螺旋运动

```csv
# test_spiral.csv
time_ms,push_mm,rotate_deg,velocity_mm_s,angular_velocity_deg_s
0,0.0,0.0,2.0,5.0
1000,2.0,5.0,2.0,5.0
2000,4.0,10.0,2.0,5.0
3000,6.0,15.0,2.0,5.0
4000,8.0,20.0,2.0,5.0
```

### 9.3 性能指标验证

- 控制周期：1kHz (±50μs)
- 轨迹跟踪误差：<0.5mm, <2°
- 延迟：<10ms
- CPU占用：<20%

## 十、与AI组对接（Day 14）

### 10.1 接口协议

AI组需要生成符合以下格式的轨迹文件：

```python
# 示例Python代码（供AI组参考）
import csv

def save_trajectory(trajectory_points, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_ms', 'push_mm', 'rotate_deg', 
                        'velocity_mm_s', 'angular_velocity_deg_s'])
      
        time_ms = 0
        for point in trajectory_points:
            writer.writerow([
                time_ms,
                point['push_mm'],
                point['rotate_deg'],
                point['velocity_mm_s'],
                point['angular_velocity_deg_s']
            ])
            time_ms += 100  # 100ms间隔
```

### 10.2 集成测试流程

1. AI组提供轨迹文件
2. 控制组验证文件格式
3. 空载测试运动轨迹
4. 硅胶模型验证
5. 记录运动数据供AI组优化

## 十一、常见问题解决

### Q1: CAN通信失败

```bash
# 检查CAN接口状态
ip link show can0

# 设置CAN接口
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 检查CAN统计
ip -details -statistics link show can0
```

### Q2: ROS2节点通信问题

```bash
# 检查节点状态
ros2 node list
ros2 node info /can_bridge

# 检查话题连接
ros2 topic info /trajectory_command -v

# 检查QoS设置
ros2 topic echo /trajectory_command --no-arr --qos-profile sensor_data
```

### Q3: 实时性问题

```bash
# 设置实时优先级
sudo chrt -f 50 ros2 run surgical_robot_control can_bridge

# CPU亲和性设置
taskset -c 2,3 ros2 launch surgical_robot_control robot_control.launch.py
```

## 十二、交付清单

### 12.1 代码交付

- [ ] ROS2功能包源码
- [ ] 编译脚本和依赖列表
- [ ] 单元测试和集成测试代码

### 12.2 文档交付

- [ ] API文档（Doxygen）
- [ ] 用户使用手册
- [ ] 故障排查指南

### 12.3 测试交付

- [ ] 测试报告（精度、延迟、稳定性）
- [ ] 测试轨迹文件集
- [ ] 性能基准数据

### 12.4 演示视频

- [ ] 环境搭建过程录屏
- [ ] 各种轨迹测试视频
- [ ] 与AI组集成演示

---

**下一步行动（控制组）**：

1. **今天**：完成WSL2和ROS2环境搭建
2. **明天**：实现轨迹播放节点
3. **第3天**：完成CAN通信测试
4. **第1周末**：实现完整控制链路
5. **第2周**：优化性能并与AI组对接