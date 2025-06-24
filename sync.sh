#!/bin/bash

# 快速同步脚本
# 用于在WSL和Windows之间同步代码

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Git 同步工具 ===${NC}"

# 检查是否有未提交的更改
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}检测到未提交的更改：${NC}"
    git status -s
    echo ""
    echo -e "${YELLOW}是否要提交这些更改？(y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "请输入提交信息："
        read -r commit_msg
        git add .
        git commit -m "$commit_msg"
    else
        echo -e "${RED}跳过提交，只同步已提交的内容${NC}"
    fi
fi

# 显示可用的远程仓库
echo ""
echo -e "${GREEN}可用的远程仓库：${NC}"
git remote -v

# 选择同步目标
echo ""
echo "请选择同步方式："
echo "1) 推送到所有远程仓库"
echo "2) 只推送到local（本地Windows）"
echo "3) 只推送到origin（GitHub/GitLab等）"
echo "4) 从远程拉取更新"
read -r choice

case $choice in
    1)
        for remote in $(git remote); do
            echo -e "${GREEN}推送到 $remote...${NC}"
            git push $remote master
        done
        ;;
    2)
        echo -e "${GREEN}推送到本地Windows仓库...${NC}"
        git push local master
        ;;
    3)
        echo -e "${GREEN}推送到远程仓库...${NC}"
        git push origin master
        ;;
    4)
        echo "从哪个远程仓库拉取？"
        select remote in $(git remote); do
            echo -e "${GREEN}从 $remote 拉取...${NC}"
            git pull $remote master
            break
        done
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}同步完成！${NC}"
echo ""

# 显示最新状态
echo -e "${GREEN}当前状态：${NC}"
git log --oneline -5
echo ""
git status 