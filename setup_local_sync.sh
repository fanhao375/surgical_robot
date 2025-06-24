#!/bin/bash

# 设置本地Git同步
# 这个脚本会在Windows文件系统中创建一个裸仓库用于同步

# 配置变量
WIN_USER="你的Windows用户名"  # 请修改为你的实际Windows用户名
BARE_REPO="/mnt/c/Users/$WIN_USER/git-repos/surgical-robot.git"
WIN_CLONE="/mnt/c/Users/$WIN_USER/Projects/surgical-robot-auto-navigation"

echo "=== 设置本地Git同步 ==="

# 1. 创建裸仓库目录
echo "1. 创建裸仓库..."
mkdir -p "$(dirname "$BARE_REPO")"
git init --bare "$BARE_REPO"

# 2. 添加裸仓库为远程仓库
echo "2. 添加本地远程仓库..."
git remote remove local 2>/dev/null || true
git remote add local "$BARE_REPO"

# 3. 推送到裸仓库
echo "3. 推送代码到裸仓库..."
git push local master

# 4. 在Windows端克隆（可选）
echo "4. 是否要在Windows端创建克隆？(y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    mkdir -p "$(dirname "$WIN_CLONE")"
    git clone "$BARE_REPO" "$WIN_CLONE"
    echo "Windows端克隆完成：$WIN_CLONE"
fi

echo ""
echo "=== 设置完成！==="
echo ""
echo "使用方法："
echo "在WSL中推送更新："
echo "  git push local master"
echo ""
echo "在Windows中拉取更新："
echo "  cd $WIN_CLONE"
echo "  git pull"
echo ""
echo "Windows路径："
echo "  裸仓库: C:\\Users\\$WIN_USER\\git-repos\\surgical-robot.git"
echo "  工作区: C:\\Users\\$WIN_USER\\Projects\\surgical-robot-auto-navigation" 