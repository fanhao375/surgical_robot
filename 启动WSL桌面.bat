@echo off
chcp 65001 > nul
title WSL桌面启动器

echo =============================================
echo           WSL桌面环境启动器
echo =============================================
echo.

echo 正在启动WSL Ubuntu-22.04...
wsl -d Ubuntu-22.04 --cd /home/haofan/surgical-robot-auto-navigation ./start_wsl_desktop.sh

pause