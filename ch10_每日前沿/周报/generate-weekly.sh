#!/bin/bash
# =============================================================================
# 自动驾驶周报生成脚本
# 用途: 为 Autopilot-Notes 项目创建新的周报文件
# 作者: 大白
# 日期: 2026-02-14
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取当前日期信息
YEAR=$(date +%Y)
WEEK=$(date +%V)
DATE_START=$(date -d "$(date +%Y)-01-01 +$(( ($(date +%W) - 1) * 7 )) days" +%Y-%m-%d 2>/dev/null || date -v"-$(($(date +%w) - 1))d" +%Y-%m-%d)
DATE_END=$(date -d "$(date +%Y)-01-01 +$(( ($(date +%W) - 1) * 7 + 6 )) days" +%Y-%m-%d 2>/dev/null || date -v"+$(echo "6-$(date +%w)" | bc)d" +%Y-%m-%d)

# 周报目录
WEEKLY_DIR="/home/admin/project/Autopilot-Notes/ch10_每日前沿/周报"
TEMPLATE_FILE="$WEEKLY_DIR/2026-W07-周报-模板.md"

# 帮助信息
show_help() {
    echo "自动驾驶周报生成工具"
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  create    创建新的周报文件 (默认)"
    echo "  check     检查周报格式"
    echo "  list      列出所有周报"
    echo "  help      显示帮助信息"
    echo ""
    echo "选项:"
    echo "  -y YEAR   指定年份 (默认: 当前年份 $YEAR)"
    echo "  -w WEEK   指定周数 (默认: 当前周数 $WEEK)"
    echo ""
    echo "示例:"
    echo "  $0                    # 创建当前周的周报"
    echo "  $0 create             # 创建当前周的周报"
    echo "  $0 create -y 2026 -w 8 # 创建2026年第8周周报"
    echo "  $0 check 2026-W07-周报.md  # 检查指定周报格式"
    echo "  $0 list               # 列出所有周报"
}

# 创建周报文件
create_weekly() {
    local target_year=${1:-$YEAR}
    local target_week=${2:-$WEEK}
    
    local filename="${target_year}-W${target_week}-周报.md"
    local filepath="$WEEKLY_DIR/$filename"
    
    echo -e "${BLUE}📅 生成周报: ${target_year}年第${target_week}周${NC}"
    echo ""
    
    # 检查目录是否存在
    if [ ! -d "$WEEKLY_DIR" ]; then
        echo -e "${RED}❌ 错误: 周报目录不存在${NC}"
        echo "   路径: $WEEKLY_DIR"
        exit 1
    fi
    
    # 检查模板文件是否存在
    if [ ! -f "$TEMPLATE_FILE" ]; then
        echo -e "${YELLOW}⚠️  警告: 模板文件不存在${NC}"
        echo "   将创建一个空的周报文件"
        TEMPLATE_FILE=""
    fi
    
    # 检查文件是否已存在
    if [ -f "$filepath" ]; then
        echo -e "${YELLOW}⚠️  周报文件已存在: $filename${NC}"
        read -p "是否覆盖? (y/N): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}已取消${NC}"
            exit 0
        fi
    fi
    
    # 创建周报文件
    if [ -n "$TEMPLATE_FILE" ] && [ -f "$TEMPLATE_FILE" ]; then
        cp "$TEMPLATE_FILE" "$filepath"
        
        # 替换模板中的日期信息
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/2026年第7周/${target_year}年第${target_week}周/g" "$filepath"
            sed -i '' "s/2月10日-2月16日/${DATE_START}-${DATE_END}/g" "$filepath"
            sed -i '' "s/2026-02-16/$(date +%Y-%m-%d)/g" "$filepath"
        else
            # Linux
            sed -i "s/2026年第7周/${target_year}年第${target_week}周/g" "$filepath"
            sed -i "s/2月10日-2月16日/${DATE_START}-${DATE_END}/g" "$filepath"
            sed -i "s/2026-02-16/$(date +%Y-%m-%d)/g" "$filepath"
        fi
        
        echo -e "${GREEN}✅ 周报创建成功!${NC}"
        echo "   文件: $filepath"
        echo ""
        echo -e "${BLUE}📝 下一步操作:${NC}"
        echo "   1. 编辑文件: $filepath"
        echo "   2. 参考 weekly-report-generator.md 收集本周资讯"
        echo "   3. 填充热点TOP5、技术趋势、公司动态等内容"
        echo ""
        echo -e "${YELLOW}💡 提示: 使用 '$0 check $filename' 检查周报格式${NC}"
    else
        # 创建空文件（带基本结构）
        cat > "$filepath" << EOF
# 自动驾驶技术周报 - ${target_year}年第${target_week}周

> 📅 **周报周期**: ${DATE_START} - ${DATE_END}
> 📝 **编辑**: 大白
> 📌 **标签**: #自动驾驶 #周报 #技术前沿

---

## 📊 本周热点 TOP5

### 1️⃣ 【热点标题】
**重要性**: ⭐⭐⭐⭐⭐  
**摘要**: ...

---

## 🔬 技术趋势分析

### 🎯 感知技术
...

### 🧠 决策规划
...

### 📜 政策法规
...

### 🏭 产业动态
...

---

## 🏢 重点关注公司动态

### Tesla
...

### Waymo
...

---

## 🔭 下周值得关注的领域

...

---

*Generated on $(date +%Y-%m-%d) | Autopilot-Notes Project*
EOF
        echo -e "${GREEN}✅ 已创建基础周报框架${NC}"
        echo "   文件: $filepath"
    fi
}

# 检查周报格式
check_weekly() {
    local filepath="$1"
    
    if [ -z "$filepath" ]; then
        # 自动查找最新的周报
        filepath=$(ls -t "$WEEKLY_DIR"/*-周报.md 2>/dev/null | head -1)
        if [ -z "$filepath" ]; then
            echo -e "${RED}❌ 错误: 未找到周报文件${NC}"
            exit 1
        fi
        echo -e "${BLUE}🔍 自动选择最新周报: $(basename "$filepath")${NC}"
    fi
    
    if [ ! -f "$filepath" ]; then
        echo -e "${RED}❌ 错误: 文件不存在${NC}"
        echo "   路径: $filepath"
        exit 1
    fi
    
    echo ""
    echo -e "${BLUE}🔍 格式检查: $(basename "$filepath")${NC}"
    echo "=========================================="
    
    local errors=0
    local warnings=0
    
    # 检查必需内容
    declare -a required_sections=(
        "本周热点 TOP5"
        "八大热门方向跟踪"
        "端到端 + VLA 大模型"
        "世界模型"
        "Robotaxi"
        "城市 NOA"
        "智驾芯片"
        "激光雷达"
        "L3 准入与法规"
        "车路云一体化"
        "重点关注公司动态"
        "下周值得关注的领域"
    )
    
    for section in "${required_sections[@]}"; do
        if grep -q "$section" "$filepath"; then
            echo -e "${GREEN}✅${NC} 包含: $section"
        else
            echo -e "${RED}❌${NC} 缺少: $section"
            ((errors++))
        fi
    done
    
    # 检查模板占位符
    echo ""
    echo -e "${BLUE}📝 内容检查:${NC}"
    
    declare -a placeholders=(
        "【热点标题"
        "示例:"
        "XX"
        "..."
        "描述"
    )
    
    for placeholder in "${placeholders[@]}"; do
        count=$(grep -c "$placeholder" "$filepath" 2>/dev/null || echo 0)
        if [ "$count" -gt 0 ]; then
            echo -e "${YELLOW}⚠️${NC} 发现 $count 处占位符: '$placeholder'"
            ((warnings++))
        fi
    done
    
    # 检查日期格式
    if grep -qE "[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日" "$filepath"; then
        echo -e "${GREEN}✅${NC} 日期格式正确"
    else
        echo -e "${YELLOW}⚠️${NC} 未检测到标准日期格式"
        ((warnings++))
    fi
    
    # 统计
    echo ""
    echo "=========================================="
    echo -e "检查完成: ${RED}$errors 个错误${NC}, ${YELLOW}$warnings 个警告${NC}"
    
    if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
        echo -e "${GREEN}🎉 周报格式检查通过!${NC}"
        return 0
    elif [ $errors -eq 0 ]; then
        echo -e "${YELLOW}⚠️  请处理警告后提交${NC}"
        return 1
    else
        echo -e "${RED}❌ 请修复错误后再提交${NC}"
        return 2
    fi
}

# 列出所有周报
list_weeklies() {
    echo -e "${BLUE}📋 周报列表${NC}"
    echo "=========================================="
    
    if [ ! -d "$WEEKLY_DIR" ]; then
        echo -e "${RED}❌ 周报目录不存在${NC}"
        exit 1
    fi
    
    local count=0
    for file in $(ls -t "$WEEKLY_DIR"/*-周报.md 2>/dev/null); do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            local size=$(du -h "$file" | cut -f1)
            local mtime=$(date -r "$file" "+%Y-%m-%d %H:%M" 2>/dev/null || stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$file" 2>/dev/null)
            echo -e "${GREEN}●${NC} $filename (${size}, $mtime)"
            ((count++))
        fi
    done
    
    if [ $count -eq 0 ]; then
        echo -e "${YELLOW}暂无周报文件${NC}"
    else
        echo ""
        echo "共 $count 篇周报"
    fi
}

# 主函数
main() {
    local cmd="${1:-create}"
    shift || true
    
    local target_year="$YEAR"
    local target_week="$WEEK"
    local filepath=""
    
    # 解析选项
    while getopts ":y:w:" opt; do
        case $opt in
            y) target_year="$OPTARG" ;;
            w) target_week="$OPTARG" ;;
            \?) echo -e "${RED}❌ 无效选项: -$OPTARG${NC}" >&2; exit 1 ;;
        esac
    done
    
    case "$cmd" in
        create)
            create_weekly "$target_year" "$target_week"
            ;;
        check)
            check_weekly "$1"
            ;;
        list)
            list_weeklies
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}❌ 未知命令: $cmd${NC}"
            echo "使用 '$0 help' 查看帮助"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
