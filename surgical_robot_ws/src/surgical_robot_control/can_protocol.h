#ifndef CAN_PROTOCOL_H
#define CAN_PROTOCOL_H

#include <cstdint>

namespace SurgicalRobot {
namespace CAN {

// CAN节点ID定义
enum NodeID : uint8_t {
    PUSH_MOTOR = 1,     // 推送电机节点ID
    ROTATE_MOTOR = 2,   // 旋转电机节点ID
    CONTROLLER = 10     // 控制器节点ID
};

// CAN消息类型定义
enum MessageType : uint16_t {
    // SDO (Service Data Object) 消息
    SDO_WRITE_REQUEST  = 0x600,  // SDO写请求 (0x600 + NodeID)
    SDO_WRITE_RESPONSE = 0x580,  // SDO写响应 (0x580 + NodeID)
    SDO_READ_REQUEST   = 0x600,  // SDO读请求
    SDO_READ_RESPONSE  = 0x580,  // SDO读响应
    
    // PDO (Process Data Object) 消息
    TPDO1 = 0x180,      // 发送PDO1 (0x180 + NodeID)
    TPDO2 = 0x280,      // 发送PDO2 (0x280 + NodeID)
    RPDO1 = 0x200,      // 接收PDO1 (0x200 + NodeID)
    RPDO2 = 0x300,      // 接收PDO2 (0x300 + NodeID)
    
    // 紧急消息
    EMERGENCY = 0x80,   // 紧急消息 (0x80 + NodeID)
    
    // 心跳消息
    HEARTBEAT = 0x700   // 心跳消息 (0x700 + NodeID)
};

// CANopen对象字典索引
enum ObjectIndex : uint16_t {
    TARGET_POSITION     = 0x6081,  // 目标位置
    TARGET_VELOCITY     = 0x6082,  // 目标速度
    ACTUAL_POSITION     = 0x6083,  // 实际位置
    ACTUAL_VELOCITY     = 0x6084,  // 实际速度
    CONTROL_WORD        = 0x6040,  // 控制字
    STATUS_WORD         = 0x6041,  // 状态字
    OPERATION_MODE      = 0x6060,  // 操作模式
    POSITION_DEMAND     = 0x6062,  // 位置需求值
    VELOCITY_DEMAND     = 0x6069,  // 速度需求值
};

// 电机操作模式
enum OperationMode : int8_t {
    POSITION_MODE = 1,  // 位置模式
    VELOCITY_MODE = 2,  // 速度模式
    TORQUE_MODE   = 3,  // 力矩模式
    HOMING_MODE   = 6   // 回零模式
};

// 控制字位定义
enum ControlWordBits : uint16_t {
    CTRL_SWITCH_ON           = 0x0001,
    CTRL_ENABLE_VOLTAGE      = 0x0002,
    CTRL_QUICK_STOP          = 0x0004,
    CTRL_ENABLE_OPERATION    = 0x0008,
    CTRL_FAULT_RESET         = 0x0080,
    CTRL_HALT                = 0x0100
};

// 状态字位定义
enum StatusWordBits : uint16_t {
    STAT_READY_TO_SWITCH_ON  = 0x0001,
    STAT_SWITCHED_ON         = 0x0002,
    STAT_OPERATION_ENABLED   = 0x0004,
    STAT_FAULT               = 0x0008,
    STAT_VOLTAGE_ENABLED     = 0x0010,
    STAT_QUICK_STOP          = 0x0020,
    STAT_SWITCH_ON_DISABLED  = 0x0040,
    STAT_WARNING             = 0x0080,
    STAT_TARGET_REACHED      = 0x0400
};

// SDO命令字
enum SDOCommand : uint8_t {
    SDO_DOWNLOAD_INIT   = 0x23,  // 下载初始化（4字节数据）
    SDO_DOWNLOAD_SEG    = 0x00,  // 下载分段
    SDO_UPLOAD_INIT     = 0x40,  // 上传初始化
    SDO_UPLOAD_SEG      = 0x60,  // 上传分段
    SDO_ABORT           = 0x80   // 中止传输
};

// 单位转换常数
constexpr double ENCODER_COUNTS_PER_MM = 1000.0;      // 编码器计数/毫米
constexpr double ENCODER_COUNTS_PER_DEGREE = 100.0;   // 编码器计数/度
constexpr double MAX_PUSH_POSITION_MM = 50.0;         // 最大推送位置
constexpr double MAX_ROTATE_ANGLE_DEG = 360.0;        // 最大旋转角度

// 安全限制
constexpr double MAX_PUSH_VELOCITY_MM_S = 10.0;       // 最大推送速度
constexpr double MAX_ANGULAR_VELOCITY_DEG_S = 30.0;   // 最大角速度

} // namespace CAN
} // namespace SurgicalRobot

#endif // CAN_PROTOCOL_H 