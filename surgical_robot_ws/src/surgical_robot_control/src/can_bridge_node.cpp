#include <rclcpp/rclcpp.hpp>
#include <cstring>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <chrono>

#include "surgical_robot_control/msg/trajectory_point.hpp"
#include "surgical_robot_control/msg/robot_state.hpp"
#include "can_protocol.h"

using namespace SurgicalRobot::CAN;

class CANBridgeNode : public rclcpp::Node
{
public:
    CANBridgeNode() : Node("can_bridge_node"), 
                      socket_fd_(-1), 
                      is_running_(false),
                      last_push_position_(0.0),
                      last_rotate_angle_(0.0)
    {
        // 声明参数
        this->declare_parameter<std::string>("can_interface", "can0");
        this->declare_parameter<bool>("enable_simulation", true);
        
        std::string can_interface = this->get_parameter("can_interface").as_string();
        simulation_mode_ = this->get_parameter("enable_simulation").as_bool();
        
        // 创建订阅器
        trajectory_sub_ = this->create_subscription<surgical_robot_control::msg::TrajectoryPoint>(
            "trajectory_command", 10,
            std::bind(&CANBridgeNode::trajectoryCallback, this, std::placeholders::_1));
        
        // 创建发布器
        robot_state_pub_ = this->create_publisher<surgical_robot_control::msg::RobotState>(
            "robot_state", 10);
        
        // 初始化CAN接口
        if (initCAN(can_interface)) {
            RCLCPP_INFO(this->get_logger(), "CAN桥接节点启动成功，接口: %s", can_interface.c_str());
            
            // 启动CAN接收线程
            is_running_ = true;
            can_thread_ = std::thread(&CANBridgeNode::canReceiveThread, this);
            
            // 创建状态发布定时器
            state_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(100),
                std::bind(&CANBridgeNode::publishRobotState, this));
        } else {
            if (simulation_mode_) {
                RCLCPP_WARN(this->get_logger(), "CAN接口初始化失败，运行在仿真模式");
                // 创建状态发布定时器（仿真模式）
                state_timer_ = this->create_wall_timer(
                    std::chrono::milliseconds(100),
                    std::bind(&CANBridgeNode::publishRobotState, this));
            } else {
                RCLCPP_ERROR(this->get_logger(), "CAN接口初始化失败，节点退出");
                return;
            }
        }
        
        // 初始化电机（如果连接成功）
        if (socket_fd_ >= 0) {
            initializeMotors();
        }
    }
    
    ~CANBridgeNode()
    {
        // 停止线程
        is_running_ = false;
        if (can_thread_.joinable()) {
            can_thread_.join();
        }
        
        // 关闭socket
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
    }

private:
    bool initCAN(const std::string& interface_name)
    {
        // 创建socket
        socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (socket_fd_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "创建CAN socket失败: %s", strerror(errno));
            return false;
        }
        
        // 检查接口是否存在
        struct ifreq ifr;
        strcpy(ifr.ifr_name, interface_name.c_str());
        if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
            RCLCPP_ERROR(this->get_logger(), "CAN接口 %s 不存在: %s", 
                        interface_name.c_str(), strerror(errno));
            close(socket_fd_);
            socket_fd_ = -1;
            return false;
        }
        
        // 绑定接口
        struct sockaddr_can addr;
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        
        if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            RCLCPP_ERROR(this->get_logger(), "绑定CAN接口失败: %s", strerror(errno));
            close(socket_fd_);
            socket_fd_ = -1;
            return false;
        }
        
        return true;
    }
    
    void initializeMotors()
    {
        RCLCPP_INFO(this->get_logger(), "初始化电机控制器...");
        
        // 设置推送电机为位置模式
        sendSDOWrite(PUSH_MOTOR, OPERATION_MODE, POSITION_MODE);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // 设置旋转电机为位置模式
        sendSDOWrite(ROTATE_MOTOR, OPERATION_MODE, POSITION_MODE);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // 启动电机
        uint16_t control_word = CTRL_ENABLE_VOLTAGE | CTRL_SWITCH_ON | CTRL_ENABLE_OPERATION;
        sendSDOWrite(PUSH_MOTOR, CONTROL_WORD, control_word);
        sendSDOWrite(ROTATE_MOTOR, CONTROL_WORD, control_word);
        
        RCLCPP_INFO(this->get_logger(), "电机初始化完成");
    }
    
    void trajectoryCallback(const surgical_robot_control::msg::TrajectoryPoint::SharedPtr msg)
    {
        RCLCPP_DEBUG(this->get_logger(), 
                    "接收轨迹点: push=%.2fmm, rotate=%.2f°", 
                    msg->push_position, msg->rotate_angle);
        
        // 安全检查
        if (!isTrajectoryPointSafe(*msg)) {
            RCLCPP_WARN(this->get_logger(), "轨迹点超出安全范围，忽略");
            return;
        }
        
        // 转换单位并发送CAN消息
        int32_t push_counts = static_cast<int32_t>(msg->push_position * ENCODER_COUNTS_PER_MM);
        int32_t rotate_counts = static_cast<int32_t>(msg->rotate_angle * ENCODER_COUNTS_PER_DEGREE);
        
        if (socket_fd_ >= 0) {
            // 发送位置命令到电机
            sendPositionCommand(PUSH_MOTOR, push_counts);
            sendPositionCommand(ROTATE_MOTOR, rotate_counts);
        } else if (simulation_mode_) {
            // 仿真模式：直接更新位置
            last_push_position_ = msg->push_position;
            last_rotate_angle_ = msg->rotate_angle;
            RCLCPP_DEBUG(this->get_logger(), 
                        "仿真模式：更新位置 push=%.2fmm, rotate=%.2f°", 
                        last_push_position_, last_rotate_angle_);
        }
    }
    
    bool isTrajectoryPointSafe(const surgical_robot_control::msg::TrajectoryPoint& point)
    {
        return (point.push_position >= 0 && point.push_position <= MAX_PUSH_POSITION_MM &&
                point.rotate_angle >= -MAX_ROTATE_ANGLE_DEG && point.rotate_angle <= MAX_ROTATE_ANGLE_DEG &&
                std::abs(point.push_velocity) <= MAX_PUSH_VELOCITY_MM_S &&
                std::abs(point.angular_velocity) <= MAX_ANGULAR_VELOCITY_DEG_S);
    }
    
    bool sendPositionCommand(uint8_t node_id, int32_t position)
    {
        return sendSDOWrite(node_id, TARGET_POSITION, position);
    }
    
    bool sendSDOWrite(uint8_t node_id, uint16_t index, int32_t data)
    {
        if (socket_fd_ < 0) return false;
        
        struct can_frame frame;
        frame.can_id = SDO_WRITE_REQUEST + node_id;
        frame.can_dlc = 8;
        
        // SDO写命令格式
        frame.data[0] = SDO_DOWNLOAD_INIT;          // 命令字
        frame.data[1] = index & 0xFF;               // 索引低字节
        frame.data[2] = (index >> 8) & 0xFF;        // 索引高字节
        frame.data[3] = 0x00;                       // 子索引
        memcpy(&frame.data[4], &data, 4);           // 数据
        
        int nbytes = write(socket_fd_, &frame, sizeof(frame));
        if (nbytes != sizeof(frame)) {
            RCLCPP_ERROR(this->get_logger(), "发送CAN帧失败: %s", strerror(errno));
            return false;
        }
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "发送SDO写命令: 节点%d, 索引0x%04X, 数据%d", 
                    node_id, index, data);
        return true;
    }
    
    void canReceiveThread()
    {
        struct can_frame frame;
        
        while (is_running_ && socket_fd_ >= 0) {
            int nbytes = read(socket_fd_, &frame, sizeof(frame));
            if (nbytes > 0) {
                processCANFrame(frame);
            } else if (nbytes < 0 && errno != EAGAIN) {
                RCLCPP_ERROR(this->get_logger(), "读取CAN消息失败: %s", strerror(errno));
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    void processCANFrame(const struct can_frame& frame)
    {
        uint8_t node_id = frame.can_id & 0x7F;  // 获取节点ID
        uint16_t function_code = frame.can_id & 0x780;
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "接收CAN消息: ID=0x%03X, 节点=%d, 功能码=0x%03X", 
                    frame.can_id, node_id, function_code);
        
        // 处理不同类型的消息
        switch (function_code) {
            case SDO_WRITE_RESPONSE:
                handleSDOResponse(node_id, frame);
                break;
            case TPDO1:
                handlePDO1(node_id, frame);
                break;
            case EMERGENCY:
                handleEmergency(node_id, frame);
                break;
            default:
                // 未知消息类型
                break;
        }
    }
    
    void handleSDOResponse(uint8_t node_id, const struct can_frame& frame)
    {
        if (frame.data[0] == SDO_ABORT) {
            uint32_t abort_code;
            memcpy(&abort_code, &frame.data[4], 4);
            RCLCPP_WARN(this->get_logger(), 
                       "节点%d SDO中止，错误码: 0x%08X", node_id, abort_code);
        } else {
            RCLCPP_DEBUG(this->get_logger(), "节点%d SDO响应正常", node_id);
        }
    }
    
    void handlePDO1(uint8_t node_id, const struct can_frame& frame)
    {
        // 解析PDO数据（假设包含位置和状态信息）
        int32_t actual_position;
        uint16_t status_word;
        
        memcpy(&actual_position, &frame.data[0], 4);
        memcpy(&status_word, &frame.data[4], 2);
        
        // 更新内部状态
        if (node_id == PUSH_MOTOR) {
            current_push_position_ = actual_position / ENCODER_COUNTS_PER_MM;
            push_status_ = status_word;
        } else if (node_id == ROTATE_MOTOR) {
            current_rotate_angle_ = actual_position / ENCODER_COUNTS_PER_DEGREE;
            rotate_status_ = status_word;
        }
    }
    
    void handleEmergency(uint8_t node_id, const struct can_frame& frame)
    {
        uint16_t error_code;
        uint8_t error_register;
        
        memcpy(&error_code, &frame.data[0], 2);
        error_register = frame.data[2];
        
        RCLCPP_ERROR(this->get_logger(), 
                    "节点%d 紧急消息: 错误码=0x%04X, 错误寄存器=0x%02X", 
                    node_id, error_code, error_register);
    }
    
    void publishRobotState()
    {
        auto state_msg = surgical_robot_control::msg::RobotState();
        
        state_msg.timestamp = this->now().seconds();
        
        if (simulation_mode_ && socket_fd_ < 0) {
            // 仿真模式
            state_msg.actual_push_position = last_push_position_;
            state_msg.actual_rotate_angle = last_rotate_angle_;
            state_msg.push_force = 0.0;
            state_msg.rotate_torque = 0.0;
            state_msg.status = 1;  // 运动中
            state_msg.error_message = "仿真模式运行";
        } else {
            // 实际模式
            state_msg.actual_push_position = current_push_position_;
            state_msg.actual_rotate_angle = current_rotate_angle_;
            state_msg.push_force = 0.0;  // 需要从传感器获取
            state_msg.rotate_torque = 0.0;
            
            // 检查电机状态
            bool push_ready = (push_status_ & STAT_OPERATION_ENABLED) != 0;
            bool rotate_ready = (rotate_status_ & STAT_OPERATION_ENABLED) != 0;
            bool has_fault = (push_status_ & STAT_FAULT) != 0 || (rotate_status_ & STAT_FAULT) != 0;
            
            if (has_fault) {
                state_msg.status = 2;  // 错误
                state_msg.error_message = "电机故障";
            } else if (push_ready && rotate_ready) {
                state_msg.status = 1;  // 运动中
                state_msg.error_message = "";
            } else {
                state_msg.status = 0;  // 空闲
                state_msg.error_message = "电机未就绪";
            }
        }
        
        robot_state_pub_->publish(state_msg);
    }
    
    // 成员变量
    rclcpp::Subscription<surgical_robot_control::msg::TrajectoryPoint>::SharedPtr trajectory_sub_;
    rclcpp::Publisher<surgical_robot_control::msg::RobotState>::SharedPtr robot_state_pub_;
    rclcpp::TimerBase::SharedPtr state_timer_;
    
    int socket_fd_;
    std::thread can_thread_;
    std::atomic<bool> is_running_;
    bool simulation_mode_;
    
    // 状态变量
    double current_push_position_ = 0.0;
    double current_rotate_angle_ = 0.0;
    uint16_t push_status_ = 0;
    uint16_t rotate_status_ = 0;
    
    // 仿真模式变量
    double last_push_position_;
    double last_rotate_angle_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CANBridgeNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 