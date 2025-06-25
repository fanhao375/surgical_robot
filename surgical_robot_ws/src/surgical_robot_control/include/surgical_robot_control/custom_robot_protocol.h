#ifndef CUSTOM_ROBOT_PROTOCOL_H
#define CUSTOM_ROBOT_PROTOCOL_H

#include "can_protocol_interface.h"
#include <map>
#include <mutex>
#include <atomic>

namespace surgical_robot {

// 根据robot协议CSV定义的自定义协议
class CustomRobotProtocol : public CANProtocolInterface {
public:
    CustomRobotProtocol();
    ~CustomRobotProtocol() override;
    
    // 实现抽象接口
    bool initialize(const std::string& can_interface) override;
    bool enableMotor(uint8_t motor_id) override;
    bool disableMotor(uint8_t motor_id) override;
    bool resetMotor(uint8_t motor_id) override;
    bool emergencyStop(uint8_t motor_id) override;
    bool sendPositionCommand(uint8_t motor_id, const MotionCommand& cmd) override;
    bool sendVelocityCommand(uint8_t motor_id, const MotionCommand& cmd) override;
    bool readMotorStatus(uint8_t motor_id, MotorStatus& status) override;
    std::string getProtocolName() const override { return "Custom Robot Protocol V1"; }
    std::string getVersion() const override { return "1.0.0"; }
    
    // 扩展接口：支持robot协议特有功能
    bool sendHandshake(uint8_t sequence);
    bool readForceData(float& grip_force, float& resistance_left, float& resistance_right);
    bool getEmergencyStatus(bool& is_emergency);
    bool resetErrors();

private:
    // ================================
    // 协议组定义 (基于CSV)
    // ================================
    
    // 组3: 握手协议
    static constexpr uint32_t HANDSHAKE_ID = 0x300;
    struct HandshakeMsg {
        uint8_t sequence;  // 0~255递增
        uint8_t reserved[7];
    } __attribute__((packed));
    
    // 组4: 急停和错误
    static constexpr uint32_t EMERGENCY_STATUS_ID = 0x400;
    static constexpr uint32_t ERROR_RESET_ID = 0x403;
    static constexpr uint32_t M3_FAULT_ID = 0x406;  // 导丝推进电机
    static constexpr uint32_t M4_FAULT_ID = 0x407;  // 导丝旋转电机
    
    // 组5: 节点状态控制
    static constexpr uint32_t BOX_STATE_CTRL_ID = 0x500;
    static constexpr uint32_t IMPACT_STATE_ID = 0x501;
    
    // 组6: 运动控制 ⭐ AI导航核心
    static constexpr uint32_t GW_GO_DATA_ID = 0x600;        // 导丝前进速度
    static constexpr uint32_t GW_ROTATE_SPEED_ID = 0x601;   // 导丝旋转速度  
    static constexpr uint32_t GW_POSITION_ID = 0x606;       // 导丝前进位置
    static constexpr uint32_t GW_ROTATE_POS_ID = 0x607;     // 导丝旋转位置
    
    // 组8: 运动信息反馈 ⭐ AI导航反馈
    static constexpr uint32_t GW_POS_FEEDBACK_ID = 0x800;   // 导丝位置反馈
    static constexpr uint32_t GW_ANGLE_FEEDBACK_ID = 0x801; // 导丝角度反馈
    
    // 组10: 传感器数据 ⭐ AI导航安全
    static constexpr uint32_t FORCE_DATA_ID = 0xA00;        // 夹紧力
    static constexpr uint32_t RESISTANCE_LEFT_ID = 0xA01;   // 阻力左
    static constexpr uint32_t RESISTANCE_RIGHT_ID = 0xA02;  // 阻力右
    
    // ================================
    // 协议数据结构
    // ================================
    
    // 运动控制命令 (组6)
    struct MotionControlMsg {
        int32_t data;           // 根据CSV: 速度(0.01mm/s或0.1°/s) 或位置(0.01mm或0.1°)  
        uint8_t reserved[4];
    } __attribute__((packed));
    
    // 位置反馈 (组8)
    struct PositionFeedbackMsg {
        int32_t position_data;  // 实际位置/角度
        uint8_t reserved[4];
    } __attribute__((packed));
    
    // 传感器数据 (组10)
    struct ForceDataMsg {
        int32_t force_value;    // 力值 (单位g)
        uint8_t reserved[4];
    } __attribute__((packed));
    
    // 急停状态 (组4)
    struct EmergencyMsg {
        uint8_t emergency_state; // 0=非急停, 1=急停
        uint8_t reserved[7];
    } __attribute__((packed));
    
    // ================================
    // AI导航专用方法
    // ================================
    
    // 导丝控制 (AI导航核心功能)
    bool sendGuidewireSpeed(int32_t forward_speed_01mm_s, int32_t rotate_speed_01deg_s);
    bool sendGuidewirePosition(int32_t forward_pos_01mm, int32_t rotate_angle_01deg);
    bool readGuidewirePosition(int32_t& actual_pos_01mm, int32_t& actual_angle_01deg);
    
    // 安全监控 (AI导航安全功能)
    bool readSafetyData(float& grip_force_g, float& resistance_left, float& resistance_right);
    bool checkMotorFaults(bool& push_fault, bool& rotate_fault);
    
    // 系统集成 (AI导航系统功能)
    bool systemHandshake();
    bool enableGuidewireControl();
    bool disableGuidewireControl();
    
    // ================================
    // 协议转换方法
    // ================================
    
    // 单位转换 (基于CSV协议定义)
    int32_t mmToProtocol(double mm) { return static_cast<int32_t>(mm * 100); }        // mm -> 0.01mm
    int32_t degToProtocol(double deg) { return static_cast<int32_t>(deg * 10); }      // deg -> 0.1°
    int32_t mmPerSecToProtocol(double mm_s) { return static_cast<int32_t>(mm_s * 100); } // mm/s -> 0.01mm/s
    int32_t degPerSecToProtocol(double deg_s) { return static_cast<int32_t>(deg_s * 10); } // deg/s -> 0.1°/s
    
    double protocolToMm(int32_t protocol) { return protocol / 100.0; }               // 0.01mm -> mm
    double protocolToDeg(int32_t protocol) { return protocol / 10.0; }               // 0.1° -> deg
    double protocolToMmPerSec(int32_t protocol) { return protocol / 100.0; }         // 0.01mm/s -> mm/s
    double protocolToDegPerSec(int32_t protocol) { return protocol / 10.0; }         // 0.1°/s -> deg/s
    
    // ================================
    // 内部状态管理
    // ================================
    
    // 握手序列号
    std::atomic<uint8_t> handshake_sequence_{0};
    
    // 缓存的状态数据
    struct CachedSensorData {
        float grip_force_g = 0.0f;
        float resistance_left = 0.0f; 
        float resistance_right = 0.0f;
        int32_t actual_position_01mm = 0;
        int32_t actual_angle_01deg = 0;
        bool emergency_state = false;
        bool push_motor_fault = false;
        bool rotate_motor_fault = false;
        std::chrono::steady_clock::time_point last_update;
    } cached_data_;
    
    std::mutex data_mutex_;
    
    // CAN消息收发
    bool sendCANMessage(uint32_t can_id, const void* data, uint8_t length);
    bool receiveCANMessage(uint32_t can_id, void* data, uint8_t& length, int timeout_ms = 100);
    void processIncomingMessages(); // 后台消息处理线程
    
    std::thread message_thread_;
    std::atomic<bool> running_{false};
};

// ================================
// AI导航专用接口扩展
// ================================

// AI导航控制器接口
class AINavigationInterface {
public:
    AINavigationInterface(std::shared_ptr<CustomRobotProtocol> protocol);
    
    // 核心导航控制
    struct NavigationCommand {
        double target_advance_mm;      // 目标推进距离 (mm)
        double target_rotation_deg;    // 目标旋转角度 (度)
        double advance_speed_mm_s;     // 推进速度 (mm/s)
        double rotation_speed_deg_s;   // 旋转速度 (度/s)
    };
    
    struct NavigationFeedback {
        double actual_advance_mm;      // 实际推进距离
        double actual_rotation_deg;    // 实际旋转角度
        double grip_force_n;           // 夹紧力 (N)
        double resistance_left;        // 左侧阻力
        double resistance_right;       // 右侧阻力
        bool is_safe;                  // 安全状态
        std::string error_message;     // 错误信息
    };
    
    // AI导航接口方法
    bool sendNavigationCommand(const NavigationCommand& cmd);
    bool getNavigationFeedback(NavigationFeedback& feedback);
    bool checkSafetyStatus(std::string& safety_message);
    bool emergencyStop();
    bool resetAndResume();
    
private:
    std::shared_ptr<CustomRobotProtocol> protocol_;
    NavigationCommand last_command_;
    std::mutex command_mutex_;
};

} // namespace surgical_robot

#endif // CUSTOM_ROBOT_PROTOCOL_H 