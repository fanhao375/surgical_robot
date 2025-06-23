#!/bin/bash

# WSL桌面启动脚本

echo "正在启动WSL桌面环境..."

# 设置环境变量
export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export XDG_SESSION_TYPE=x11

# 创建运行时目录
mkdir -p $XDG_RUNTIME_DIR

# 启动D-Bus会话
if [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
    echo "启动D-Bus会话..."
    eval $(dbus-launch --sh-syntax --exit-with-session)
    export DBUS_SESSION_BUS_ADDRESS
fi

# 启动XFCE桌面环境
echo "启动XFCE桌面..."
nohup xfce4-session > /dev/null 2>&1 &

# 等待一下让桌面启动
sleep 3

# 启动文件管理器
echo "启动文件管理器..."
nohup thunar > /dev/null 2>&1 &

echo "WSL桌面已启动！"
echo "如果看到桌面窗口，说明启动成功。"
echo "按任意键关闭此窗口..."
read -n 1 