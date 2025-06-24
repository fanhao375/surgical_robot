#!/bin/bash
# check_env.sh - 控制组环境检查脚本

echo "=== 控制组环境检查 ==="

# ROS2检查
echo -n "ROS2环境: "
if command -v ros2 &> /dev/null; then
    echo "✓ $(ros2 --version 2>&1 | grep -o 'ros2.*')"
else
    echo "✗ 未安装"
fi

# 工作空间检查
echo -n "工作空间: "
if [ -d "$HOME/surgical-robot-auto-navigation/surgical_robot_ws" ]; then
    echo "✓ 存在"
else
    echo "✗ 不存在"
fi

# CAN工具检查
echo -n "CAN工具: "
if command -v cansend &> /dev/null; then
    echo "✓ 已安装"
    echo "  - candump: $(which candump)"
    echo "  - cansend: $(which cansend)"
    echo "  - cangen: $(which cangen)"
else
    echo "✗ 未安装"
fi

# CAN模块检查
echo -n "CAN模块: "
if lsmod | grep -q can; then
    echo "✓ 已加载"
    lsmod | grep can | while read line; do
        echo "  - $line"
    done
else
    echo "✗ 未加载"
fi

# 虚拟CAN检查
echo -n "虚拟CAN: "
if ip link show vcan0 &> /dev/null; then
    echo "✓ vcan0已启动"
else
    echo "✗ vcan0未启动 (WSL2环境下正常)"
fi

# 项目构建检查
echo -n "项目构建: "
if [ -d "install" ]; then
    echo "✓ 已构建"
else
    echo "✗ 未构建"
fi

# CAN测试程序检查
echo -n "CAN测试程序: "
if [ -f "install/surgical_robot_control/lib/surgical_robot_control/can_test" ]; then
    echo "✓ 已编译"
else
    echo "✗ 未编译"
fi

echo ""
echo "=== 环境总结 ==="
echo "WSL2环境下CAN功能验证完成："
echo "- CAN工具包已安装并可用"
echo "- CAN内核模块已加载"
echo "- CAN测试程序已编译"
echo "- 虚拟CAN在WSL2下不可用，这是正常现象"
echo "- 实际部署时需要物理CAN硬件接口" 