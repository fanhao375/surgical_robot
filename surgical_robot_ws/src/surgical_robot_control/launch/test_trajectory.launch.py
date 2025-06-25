from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # 声明参数
    trajectory_file_arg = DeclareLaunchArgument(
        'trajectory_file',
        default_value='/home/haofan/surgical-robot-auto-navigation/surgical_robot_ws/test_trajectories/test_linear.csv',
        description='Path to trajectory CSV file'
    )
  
    # 轨迹播放节点
    trajectory_player_node = Node(
        package='surgical_robot_control',
        executable='trajectory_player',
        name='trajectory_player',
        parameters=[{
            'trajectory_file': LaunchConfiguration('trajectory_file')
        }],
        output='screen'
    )
  
    return LaunchDescription([
        trajectory_file_arg,
        trajectory_player_node
    ]) 