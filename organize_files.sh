#!/bin/bash

# 手术机器人自动导航项目 - 文件整理脚本

echo "开始整理项目文件..."

# 创建目录结构
mkdir -p docs/overview
mkdir -p docs/ai-team
mkdir -p docs/control-team
mkdir -p docs/progress
mkdir -p src/ai
mkdir -p src/control
mkdir -p config
mkdir -p tests

# 移动文档文件
echo "移动项目文档..."
if [ -f "手术机器人自动导航项目技术大纲.md" ]; then
    mv "手术机器人自动导航项目技术大纲.md" "docs/overview/"
fi

echo "移动AI组文档..."
if [ -f "AI组-第一周任务清单.md" ]; then
    mv "AI组-第一周任务清单.md" "docs/ai-team/"
fi
if [ -f "AI组-医学影像分割技术方案.md" ]; then
    mv "AI组-医学影像分割技术方案.md" "docs/ai-team/"
fi

echo "移动控制组文档..."
if [ -f "控制组-第一周任务清单.md" ]; then
    mv "控制组-第一周任务清单.md" "docs/control-team/"
fi
if [ -f "控制组-ROS2与CAN集成技术方案.md" ]; then
    mv "控制组-ROS2与CAN集成技术方案.md" "docs/control-team/"
fi

echo "移动进度报告..."
if [ -f "第一天总结报告.md" ]; then
    mv "第一天总结报告.md" "docs/progress/"
fi

echo "移动配置文件..."
if [ -f "markdown-preview.css" ]; then
    mv "markdown-preview.css" "config/"
fi

# 保留文件结构说明在根目录
echo "文件整理完成！"
echo ""
echo "新的目录结构："
tree -L 3 2>/dev/null || find . -type d -not -path '*/\.*' | sed 's/[^/]*\//  /g' | sort 