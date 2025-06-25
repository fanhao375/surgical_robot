#include <rclcpp/rclcpp.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include "surgical_robot_control/msg/trajectory_point.hpp"

struct TrajectoryData {
    double time_ms;
    double push_mm;
    double rotate_deg;
    double velocity_mm_s;
    double angular_velocity_deg_s;
};

class TrajectoryPlayer : public rclcpp::Node
{
public:
    TrajectoryPlayer() : Node("trajectory_player"), current_index_(0)
    {
        // 声明参数
        this->declare_parameter<std::string>("trajectory_file", "");
        
        // 创建发布器
        trajectory_pub_ = this->create_publisher<surgical_robot_control::msg::TrajectoryPoint>(
            "trajectory_command", 10);
        
        // 加载轨迹
        std::string filename = this->get_parameter("trajectory_file").as_string();
        if (!loadTrajectory(filename)) {
            RCLCPP_ERROR(this->get_logger(), "无法加载轨迹文件: %s", filename.c_str());
            return;
        }
        
        // 创建定时器 (10ms)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&TrajectoryPlayer::timerCallback, this));
        
        start_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "开始播放轨迹，共 %zu 个点", trajectory_.size());
    }

private:
    bool loadTrajectory(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        // 跳过标题行
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            TrajectoryData point;
            char comma;
            
            ss >> point.time_ms >> comma
               >> point.push_mm >> comma
               >> point.rotate_deg >> comma
               >> point.velocity_mm_s >> comma
               >> point.angular_velocity_deg_s;
            
            trajectory_.push_back(point);
        }
        
        return !trajectory_.empty();
    }
    
    void timerCallback()
    {
        if (current_index_ >= trajectory_.size()) {
            RCLCPP_INFO_ONCE(this->get_logger(), "轨迹播放完成");
            return;
        }
        
        auto elapsed = (this->now() - start_time_).seconds() * 1000.0; // ms
        
        // 找到当前应该发送的轨迹点
        while (current_index_ < trajectory_.size() && 
               trajectory_[current_index_].time_ms <= elapsed) {
            
            auto msg = surgical_robot_control::msg::TrajectoryPoint();
            msg.timestamp = this->now().seconds();
            msg.push_position = trajectory_[current_index_].push_mm;
            msg.rotate_angle = trajectory_[current_index_].rotate_deg;
            msg.push_velocity = trajectory_[current_index_].velocity_mm_s;
            msg.angular_velocity = trajectory_[current_index_].angular_velocity_deg_s;
            
            trajectory_pub_->publish(msg);
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 100,
                "发送轨迹点 [%zu/%zu]: push=%.2fmm, rotate=%.2f°",
                current_index_ + 1, trajectory_.size(),
                msg.push_position, msg.rotate_angle);
            
            current_index_++;
        }
    }
    
    rclcpp::Publisher<surgical_robot_control::msg::TrajectoryPoint>::SharedPtr trajectory_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<TrajectoryData> trajectory_;
    size_t current_index_;
    rclcpp::Time start_time_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrajectoryPlayer>());
    rclcpp::shutdown();
    return 0;
} 