from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # 声明参数
    trajectory_file_arg = DeclareLaunchArgument(
        'trajectory_file',
        default_value=os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', 
            'test_trajectories', 
            'test_linear.csv'
        ),
        description='Path to trajectory CSV file'
    )
    
    simulation_mode_arg = DeclareLaunchArgument(
        'enable_simulation',
        default_value='true',
        description='Enable simulation mode for CAN interface'
    )
    
    can_interface_arg = DeclareLaunchArgument(
        'can_interface',
        default_value='can0',
        description='CAN interface name'
    )
    
    # CAN桥接节点
    can_bridge_node = Node(
        package='surgical_robot_control',
        executable='can_bridge_node',
        name='can_bridge_node',
        parameters=[{
            'can_interface': LaunchConfiguration('can_interface'),
            'enable_simulation': LaunchConfiguration('enable_simulation')
        }],
        output='screen'
    )
    
    # 轨迹播放节点（延迟2秒启动，让CAN桥接节点先初始化）
    trajectory_player_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='surgical_robot_control',
                executable='trajectory_player',
                name='trajectory_player',
                parameters=[{
                    'trajectory_file': LaunchConfiguration('trajectory_file')
                }],
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        trajectory_file_arg,
        simulation_mode_arg,
        can_interface_arg,
        can_bridge_node,
        trajectory_player_node
    ]) 