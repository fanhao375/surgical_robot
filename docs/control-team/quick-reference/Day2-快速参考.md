# 控制组 Day 2 - 快速参考指南

## 🚀 快速开始

### 1. 创建工作空间（第一步）
```bash
mkdir -p ~/surgical_robot_ws/src
cd ~/surgical_robot_ws
```

### 2. 创建功能包
```bash
cd src
ros2 pkg create --build-type ament_cmake surgical_robot_control \
    --dependencies rclcpp std_msgs geometry_msgs
```

### 3. Hello World节点代码
保存到 `surgical_robot_control/src/hello_control.cpp`：

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class HelloControl : public rclcpp::Node
{
public:
    HelloControl() : Node("hello_control")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("control_status", 10);
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&HelloControl::timer_callback, this));
        RCLCPP_INFO(this->get_logger(), "控制节点已启动");
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "控制系统正常运行";
        publisher_->publish(message);
    }
  
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HelloControl>());
    rclcpp::shutdown();
    return 0;
}
```

### 4. 修改CMakeLists.txt
在 `find_package` 之后添加：

```cmake
add_executable(hello_control src/hello_control.cpp)
ament_target_dependencies(hello_control rclcpp std_msgs)

install(TARGETS
    hello_control
    DESTINATION lib/${PROJECT_NAME}
)
```

### 5. 构建和运行
```bash
# 构建
cd ~/surgical_robot_ws
colcon build --packages-select surgical_robot_control

# 运行（新终端）
source ~/surgical_robot_ws/install/setup.bash
ros2 run surgical_robot_control hello_control

# 查看话题（另一个新终端）
source ~/surgical_robot_ws/install/setup.bash
ros2 topic list
ros2 topic echo /control_status
```

## 🛠️ 常用命令

```bash
# 查看所有节点
ros2 node list

# 查看节点信息
ros2 node info /hello_control

# 查看所有话题
ros2 topic list

# 查看话题信息
ros2 topic info /control_status

# 实时查看话题数据
ros2 topic echo /control_status

# 查看话题发布频率
ros2 topic hz /control_status
```

## ⚠️ 常见问题

1. **找不到包**
   ```bash
   source ~/surgical_robot_ws/install/setup.bash
   ```

2. **构建失败**
   - 检查CMakeLists.txt语法
   - 确保依赖都已声明
   - 清理后重新构建：
   ```bash
   rm -rf build install log
   colcon build
   ```

3. **节点无响应**
   - 检查ROS_DOMAIN_ID是否一致
   - 使用`ros2 doctor`诊断

## ✅ Day 2 验收清单

- [ ] 工作空间创建成功
- [ ] surgical_robot_control包创建成功
- [ ] hello_control节点编译通过
- [ ] 节点运行并能看到"控制节点已启动"
- [ ] 能够用ros2 topic echo看到消息
- [ ] 消息发布频率为1Hz

## 📚 补充学习资源

- [ROS2官方教程](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS2中文社区](https://www.guyuehome.com/)
- 调试技巧：使用`RCLCPP_INFO`打印日志

---
💡 **提示**：每个步骤完成后都运行一次验证，确保环境正确！ 