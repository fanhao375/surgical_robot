#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>

class CANInterface {
public:
    CANInterface(const std::string& interface) : interface_name_(interface) {}
  
    bool init() {
        // 创建socket
        socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (socket_fd_ < 0) {
            std::cerr << "创建CAN socket失败: " << strerror(errno) << std::endl;
            return false;
        }
      
        // 检查接口是否存在
        struct ifreq ifr;
        strcpy(ifr.ifr_name, interface_name_.c_str());
        if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
            std::cerr << "CAN接口 " << interface_name_ << " 不存在: " << strerror(errno) << std::endl;
            std::cerr << "注意：在WSL2环境下，虚拟CAN接口可能不可用" << std::endl;
            close(socket_fd_);
            return false;
        }
      
        // 绑定接口
        struct sockaddr_can addr;
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
      
        if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            std::cerr << "绑定CAN接口失败: " << strerror(errno) << std::endl;
            close(socket_fd_);
            return false;
        }
      
        std::cout << "CAN接口 " << interface_name_ << " 初始化成功" << std::endl;
        return true;
    }
  
    bool sendPositionCommand(uint32_t node_id, int32_t position) {
        if (socket_fd_ < 0) {
            std::cerr << "CAN接口未初始化" << std::endl;
            return false;
        }
        
        struct can_frame frame;
        frame.can_id = 0x600 + node_id;  // SDO命令
        frame.can_dlc = 8;
      
        // SDO下载命令 - 写入目标位置 (0x6081)
        frame.data[0] = 0x23;  // 命令字
        frame.data[1] = 0x81;  // 索引低字节
        frame.data[2] = 0x60;  // 索引高字节
        frame.data[3] = 0x00;  // 子索引
        memcpy(&frame.data[4], &position, 4);  // 位置数据
      
        int nbytes = write(socket_fd_, &frame, sizeof(frame));
        if (nbytes != sizeof(frame)) {
            std::cerr << "发送CAN帧失败: " << strerror(errno) << std::endl;
            return false;
        }
      
        std::cout << "发送位置命令到节点 " << node_id 
                  << ": " << position << " counts" << std::endl;
        return true;
    }
    
    void testCanUtils() {
        std::cout << "\n=== CAN工具测试 ===" << std::endl;
        std::cout << "CAN工具已安装：" << std::endl;
        system("which candump");
        system("which cansend");
        system("which cangen");
        
        std::cout << "\nCAN模块状态：" << std::endl;
        system("lsmod | grep can");
        
        std::cout << "\n注意：在WSL2环境下，需要物理CAN接口才能进行实际通信测试" << std::endl;
        std::cout << "如需测试虚拟CAN，建议在原生Linux环境或Docker容器中运行" << std::endl;
    }
  
    ~CANInterface() {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
    }
  
private:
    std::string interface_name_;
    int socket_fd_ = -1;
};

int main() {
    std::cout << "=== CAN接口测试程序 ===" << std::endl;
    
    // 先测试CAN工具
    CANInterface can("can0");  // 使用假设的CAN接口名
    can.testCanUtils();
    
    std::cout << "\n=== CAN接口初始化测试 ===" << std::endl;
    if (!can.init()) {
        std::cout << "CAN接口初始化失败，这在WSL2环境下是正常的" << std::endl;
        std::cout << "实际部署时，请确保有可用的CAN硬件接口" << std::endl;
        return 0;  // 不返回错误，因为这在WSL2下是预期的
    }
  
    // 如果初始化成功，测试发送命令
    std::cout << "\n=== CAN命令发送测试 ===" << std::endl;
    can.sendPositionCommand(1, 1000);  // 节点1，位置1000
    can.sendPositionCommand(2, 500);   // 节点2，位置500
  
    return 0;
} 