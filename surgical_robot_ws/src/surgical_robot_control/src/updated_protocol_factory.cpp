#include "can_protocol_interface.h"
#include "canopen_protocol.h"
#include "custom_robot_protocol.h"

namespace surgical_robot {

// 扩展协议类型枚举
enum class ProtocolType {
    CANOPEN,        // 标准CANopen协议 (电机驱动器)
    CUSTOM_ROBOT    // 自定义机器人协议 (现有下位机)
};

std::unique_ptr<CANProtocolInterface> CANProtocolFactory::create(ProtocolType type) {
    switch (type) {
        case ProtocolType::CANOPEN:
            return std::make_unique<CANopenProtocol>();
            
        case ProtocolType::CUSTOM_ROBOT:
            return std::make_unique<CustomRobotProtocol>();
            
        default:
            return nullptr;
    }
}

std::vector<std::string> CANProtocolFactory::getAvailableProtocols() {
    return {
        "CANopen",      // 用于电机驱动器直接控制
        "CustomRobot"   // 用于现有机器人下位机
    };
}

// 协议自动检测 (可选功能)
ProtocolType CANProtocolFactory::detectProtocol(const std::string& can_interface) {
    // 尝试检测握手消息来判断协议类型
    // 这里简化实现，实际可以通过CAN ID范围或特定消息来检测
    
    // 检测自定义协议的握手消息 (0x300)
    if (checkForCustomHandshake(can_interface)) {
        return ProtocolType::CUSTOM_ROBOT;
    }
    
    // 检测CANopen心跳消息 (0x700+NodeID)
    if (checkForCANopenHeartbeat(can_interface)) {
        return ProtocolType::CANOPEN;
    }
    
    // 默认返回自定义协议
    return ProtocolType::CUSTOM_ROBOT;
}

} // namespace surgical_robot

// ================================
// AI导航系统使用示例
// ================================

#include <rclcpp/rclcpp.hpp>
#include "custom_robot_protocol.h"

class AINavigationNode : public rclcpp::Node {
public:
    AINavigationNode() : Node("ai_navigation_node") {
        // 创建自定义协议实例
        auto protocol = std::make_shared<CustomRobotProtocol>();
        
        // 初始化协议
        if (!protocol->initialize("can0")) {
            RCLCPP_ERROR(this->get_logger(), "协议初始化失败");
            return;
        }
        
        // 创建AI导航接口
        ai_interface_ = std::make_unique<AINavigationInterface>(protocol);
        
        // 系统握手
        if (!protocol->systemHandshake()) {
            RCLCPP_ERROR(this->get_logger(), "系统握手失败");
            return;
        }
        
        // 启用导丝控制
        if (!protocol->enableGuidewireControl()) {
            RCLCPP_ERROR(this->get_logger(), "导丝控制启用失败"); 
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "AI导航系统初始化成功");
        
        // 创建控制循环
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // 100Hz
            std::bind(&AINavigationNode::controlLoop, this));
            
        // 创建安全监控
        safety_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),  // 20Hz
            std::bind(&AINavigationNode::safetyCheck, this));
    }
    
private:
    void controlLoop() {
        // AI路径规划生成的命令 (示例)
        AINavigationInterface::NavigationCommand cmd;
        
        // 从AI模块获取导航命令
        cmd = getAINavigationCommand();
        
        // 发送导航命令
        if (!ai_interface_->sendNavigationCommand(cmd)) {
            RCLCPP_WARN(this->get_logger(), "导航命令发送失败");
            return;
        }
        
        // 获取反馈
        AINavigationInterface::NavigationFeedback feedback;
        if (ai_interface_->getNavigationFeedback(feedback)) {
            // 更新AI模块的反馈信息
            updateAIFeedback(feedback);
            
            // 日志记录 (调试用)
            RCLCPP_DEBUG(this->get_logger(), 
                "位置: %.2fmm, 角度: %.1f°, 力: %.1fN", 
                feedback.actual_advance_mm,
                feedback.actual_rotation_deg, 
                feedback.grip_force_n);
        }
    }
    
    void safetyCheck() {
        std::string safety_message;
        if (!ai_interface_->checkSafetyStatus(safety_message)) {
            RCLCPP_ERROR(this->get_logger(), "安全检查失败: %s", safety_message.c_str());
            
            // 触发紧急停止
            ai_interface_->emergencyStop();
            
            // 通知AI模块暂停
            notifyAIEmergencyStop();
        }
    }
    
    // AI接口方法 (需要与AI组对接)
    AINavigationInterface::NavigationCommand getAINavigationCommand() {
        // 这里应该调用AI模块的接口
        // 示例：固定轨迹
        static double advance = 0.0;
        static double rotation = 0.0;
        
        AINavigationInterface::NavigationCommand cmd;
        cmd.target_advance_mm = advance += 0.1;  // 每次前进0.1mm
        cmd.target_rotation_deg = rotation += 1.0; // 每次旋转1度
        cmd.advance_speed_mm_s = 2.0;
        cmd.rotation_speed_deg_s = 30.0;
        
        return cmd;
    }
    
    void updateAIFeedback(const AINavigationInterface::NavigationFeedback& feedback) {
        // 将反馈信息传递给AI模块
        // 包括位置、力、阻力等信息用于路径优化
    }
    
    void notifyAIEmergencyStop() {
        // 通知AI模块进入紧急停止状态
        RCLCPP_ERROR(this->get_logger(), "AI导航紧急停止");
    }
    
    std::unique_ptr<AINavigationInterface> ai_interface_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr safety_timer_;
};

// ================================
// 启动配置示例
// ================================

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    // 根据参数选择协议
    std::string protocol_type = "CustomRobot";  // 默认使用自定义协议
    
    if (argc > 1) {
        protocol_type = argv[1];
    }
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "使用协议: %s", protocol_type.c_str());
    
    // 启动AI导航节点
    auto node = std::make_shared<AINavigationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}

// ================================
// 配置切换示例
// ================================

// 启动脚本示例
/*
#!/bin/bash
# start_ai_navigation.sh

# 设置CAN接口
sudo ip link set can0 type can bitrate 1000000
sudo ip link set up can0

# 根据硬件类型选择协议
if [ "$1" == "direct" ]; then
    echo "启动直接电机控制模式 (CANopen)"
    ros2 run surgical_robot_control ai_navigation_node CANopen \
        --ros-args --params-file config/canopen_config.yaml
else
    echo "启动机器人下位机模式 (CustomRobot)"
    ros2 run surgical_robot_control ai_navigation_node CustomRobot \
        --ros-args --params-file config/custom_robot_config.yaml
fi
*/ 