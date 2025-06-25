/* 包含ROS2的C++客户端库头文件，提供节点、发布器、订阅器等核心功能 */
#include <rclcpp/rclcpp.hpp>
/* 包含标准字符串消息类型，用于发布文本消息 */
#include <std_msgs/msg/string.hpp>

/* 定义HelloControl类，继承自rclcpp::Node基类 */
/* 这使得该类具备了ROS2节点的所有基本功能 */
class HelloControl : public rclcpp::Node
{
public:
    /* 构造函数，通过初始化列表调用父类Node的构造函数 */
    /* "hello_control"是节点的名称，在ROS2网络中唯一标识此节点 */
    HelloControl() : Node("hello_control")
    {
        /* 创建一个发布器对象，用于发布std_msgs::msg::String类型的消息 */
        /* "control_status"是话题名称，10是消息队列的大小 */
        /* 队列大小决定了在订阅者处理消息较慢时能够缓存的消息数量 */
        publisher_ = this->create_publisher<std_msgs::msg::String>("control_status", 10);
        
        /* 创建一个墙钟定时器，每隔1秒触发一次回调函数 */
        /* std::chrono::seconds(1)指定时间间隔为1秒 */
        /* std::bind将timer_callback成员函数绑定为定时器的回调函数 */
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&HelloControl::timer_callback, this));
            
        /* 使用ROS2日志系统输出信息级别的日志 */
        /* this->get_logger()获取当前节点的日志器对象 */
        RCLCPP_INFO(this->get_logger(), "控制节点已启动（测试）");
    }

private:
    /* 定时器回调函数，每当定时器触发时被调用 */
    /* 该函数负责创建消息并发布到话题 */
    void timer_callback()
    {
        /* 创建一个std_msgs::msg::String类型的消息对象 */
        /* auto关键字让编译器自动推断变量类型 */
        auto message = std_msgs::msg::String();
        
        /* 设置消息的数据内容为中文字符串 */
        /* data是String消息类型的唯一字段 */
        message.data = "控制系统正常运行";
        
        /* 通过发布器将消息发布到"control_status"话题 */
        /* 所有订阅此话题的节点都会收到这条消息 */
        publisher_->publish(message);
    }
  
    /* 声明发布器的智能指针成员变量 */
    /* SharedPtr是ROS2推荐的智能指针类型，自动管理内存 */
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    
    /* 声明定时器的智能指针成员变量 */
    /* TimerBase是所有定时器类型的基类 */
    rclcpp::TimerBase::SharedPtr timer_;
};

/* 主函数 - 程序的入口点 */
/* argc: 命令行参数的数量, argv: 命令行参数的字符串数组 */
int main(int argc, char * argv[])
{
    /* 初始化ROS2运行时环境 */
    /* 必须在使用任何ROS2功能之前调用 */
    /* argc和argv用于处理ROS2特定的命令行参数 */
    rclcpp::init(argc, argv);
    
    /* 创建HelloControl节点的智能指针实例并开始运行 */
    /* rclcpp::spin()会阻塞当前线程，持续处理回调函数、定时器等事件 */
    /* std::make_shared创建智能指针，自动管理对象生命周期 */
    rclcpp::spin(std::make_shared<HelloControl>());
    
    /* 清理ROS2运行时环境，释放所有资源 */
    /* 通常在程序退出前调用 */
    rclcpp::shutdown();
    
    /* 返回0表示程序正常结束 */
    return 0;
} 