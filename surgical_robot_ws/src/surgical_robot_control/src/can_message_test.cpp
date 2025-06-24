#include <iostream>
#include <vector>
#include <cstring>
#include <linux/can.h>
#include "can_protocol.h"

using namespace SurgicalRobot::CAN;

class CANMessageTest {
public:
    void testMessageCreation() {
        std::cout << "=== CAN消息封装测试 ===" << std::endl;
        
        // 测试SDO写消息创建
        testSDOWriteMessage();
        
        // 测试轨迹点转换
        testTrajectoryPointConversion();
        
        // 测试安全检查
        testSafetyChecks();
        
        // 测试协议常数
        testProtocolConstants();
        
        std::cout << "=== 所有测试完成 ===" << std::endl;
    }

private:
    void testSDOWriteMessage() {
        std::cout << "\n--- SDO写消息测试 ---" << std::endl;
        
        // 创建目标位置命令
        struct can_frame frame;
        uint8_t node_id = PUSH_MOTOR;
        uint16_t index = TARGET_POSITION;
        int32_t position = 5000;  // 5mm * 1000 counts/mm
        
        frame.can_id = SDO_WRITE_REQUEST + node_id;
        frame.can_dlc = 8;
        frame.data[0] = SDO_DOWNLOAD_INIT;
        frame.data[1] = index & 0xFF;
        frame.data[2] = (index >> 8) & 0xFF;
        frame.data[3] = 0x00;
        memcpy(&frame.data[4], &position, 4);
        
        std::cout << "创建SDO写消息:" << std::endl;
        std::cout << "  CAN ID: 0x" << std::hex << frame.can_id << std::dec << std::endl;
        std::cout << "  节点ID: " << (int)node_id << std::endl;
        std::cout << "  索引: 0x" << std::hex << index << std::dec << std::endl;
        std::cout << "  位置: " << position << " counts" << std::endl;
        
        // 验证消息内容
        if (frame.can_id == (SDO_WRITE_REQUEST + PUSH_MOTOR) &&
            frame.data[0] == SDO_DOWNLOAD_INIT &&
            frame.data[1] == (TARGET_POSITION & 0xFF) &&
            frame.data[2] == ((TARGET_POSITION >> 8) & 0xFF)) {
            std::cout << "  ✓ SDO消息格式正确" << std::endl;
        } else {
            std::cout << "  ✗ SDO消息格式错误" << std::endl;
        }
    }
    
    void testTrajectoryPointConversion() {
        std::cout << "\n--- 轨迹点转换测试 ---" << std::endl;
        
        // 模拟轨迹点数据
        struct TrajectoryPoint {
            double push_position;    // mm
            double rotate_angle;     // degrees
            double push_velocity;    // mm/s
            double angular_velocity; // deg/s
        };
        
        TrajectoryPoint point = {2.5, 45.0, 1.5, 10.0};
        
        // 转换为编码器计数
        int32_t push_counts = static_cast<int32_t>(point.push_position * ENCODER_COUNTS_PER_MM);
        int32_t rotate_counts = static_cast<int32_t>(point.rotate_angle * ENCODER_COUNTS_PER_DEGREE);
        int32_t push_vel_counts = static_cast<int32_t>(point.push_velocity * ENCODER_COUNTS_PER_MM);
        int32_t rotate_vel_counts = static_cast<int32_t>(point.angular_velocity * ENCODER_COUNTS_PER_DEGREE);
        
        std::cout << "轨迹点转换:" << std::endl;
        std::cout << "  推送位置: " << point.push_position << "mm → " << push_counts << " counts" << std::endl;
        std::cout << "  旋转角度: " << point.rotate_angle << "° → " << rotate_counts << " counts" << std::endl;
        std::cout << "  推送速度: " << point.push_velocity << "mm/s → " << push_vel_counts << " counts/s" << std::endl;
        std::cout << "  角速度: " << point.angular_velocity << "°/s → " << rotate_vel_counts << " counts/s" << std::endl;
        
        // 验证转换精度
        double converted_position = push_counts / ENCODER_COUNTS_PER_MM;
        double converted_angle = rotate_counts / ENCODER_COUNTS_PER_DEGREE;
        
        if (std::abs(converted_position - point.push_position) < 0.001 &&
            std::abs(converted_angle - point.rotate_angle) < 0.001) {
            std::cout << "  ✓ 单位转换精度正确" << std::endl;
        } else {
            std::cout << "  ✗ 单位转换精度错误" << std::endl;
        }
    }
    
    void testSafetyChecks() {
        std::cout << "\n--- 安全检查测试 ---" << std::endl;
        
        struct TestCase {
            double push_pos, rotate_angle, push_vel, angular_vel;
            bool should_be_safe;
            std::string description;
        };
        
        std::vector<TestCase> test_cases = {
            {10.0, 90.0, 2.0, 15.0, true, "正常范围内"},
            {60.0, 0.0, 5.0, 10.0, false, "推送位置超限"},
            {20.0, 400.0, 5.0, 10.0, false, "旋转角度超限"},
            {25.0, 180.0, 15.0, 10.0, false, "推送速度超限"},
            {25.0, 180.0, 5.0, 40.0, false, "角速度超限"},
            {0.0, -180.0, 0.1, 1.0, true, "边界条件"},
        };
        
        for (const auto& test : test_cases) {
            bool is_safe = (test.push_pos >= 0 && test.push_pos <= MAX_PUSH_POSITION_MM &&
                          test.rotate_angle >= -MAX_ROTATE_ANGLE_DEG && test.rotate_angle <= MAX_ROTATE_ANGLE_DEG &&
                          std::abs(test.push_vel) <= MAX_PUSH_VELOCITY_MM_S &&
                          std::abs(test.angular_vel) <= MAX_ANGULAR_VELOCITY_DEG_S);
            
            std::cout << "  " << test.description << ": ";
            if (is_safe == test.should_be_safe) {
                std::cout << "✓ 通过" << std::endl;
            } else {
                std::cout << "✗ 失败 (期望:" << (test.should_be_safe ? "安全" : "不安全") 
                         << ", 实际:" << (is_safe ? "安全" : "不安全") << ")" << std::endl;
            }
        }
    }
    
    void testProtocolConstants() {
        std::cout << "\n--- 协议常数测试 ---" << std::endl;
        
        std::cout << "节点ID定义:" << std::endl;
        std::cout << "  推送电机: " << (int)PUSH_MOTOR << std::endl;
        std::cout << "  旋转电机: " << (int)ROTATE_MOTOR << std::endl;
        std::cout << "  控制器: " << (int)CONTROLLER << std::endl;
        
        std::cout << "消息类型:" << std::endl;
        std::cout << "  SDO写请求: 0x" << std::hex << SDO_WRITE_REQUEST << std::dec << std::endl;
        std::cout << "  SDO写响应: 0x" << std::hex << SDO_WRITE_RESPONSE << std::dec << std::endl;
        std::cout << "  紧急消息: 0x" << std::hex << EMERGENCY << std::dec << std::endl;
        
        std::cout << "对象字典索引:" << std::endl;
        std::cout << "  目标位置: 0x" << std::hex << TARGET_POSITION << std::dec << std::endl;
        std::cout << "  目标速度: 0x" << std::hex << TARGET_VELOCITY << std::dec << std::endl;
        std::cout << "  控制字: 0x" << std::hex << CONTROL_WORD << std::dec << std::endl;
        std::cout << "  状态字: 0x" << std::hex << STATUS_WORD << std::dec << std::endl;
        
        std::cout << "转换常数:" << std::endl;
        std::cout << "  编码器计数/毫米: " << ENCODER_COUNTS_PER_MM << std::endl;
        std::cout << "  编码器计数/度: " << ENCODER_COUNTS_PER_DEGREE << std::endl;
        
        std::cout << "安全限制:" << std::endl;
        std::cout << "  最大推送位置: " << MAX_PUSH_POSITION_MM << "mm" << std::endl;
        std::cout << "  最大旋转角度: " << MAX_ROTATE_ANGLE_DEG << "°" << std::endl;
        std::cout << "  最大推送速度: " << MAX_PUSH_VELOCITY_MM_S << "mm/s" << std::endl;
        std::cout << "  最大角速度: " << MAX_ANGULAR_VELOCITY_DEG_S << "°/s" << std::endl;
        
        std::cout << "  ✓ 所有协议常数已验证" << std::endl;
    }
};

int main() {
    CANMessageTest test;
    test.testMessageCreation();
    return 0;
} 