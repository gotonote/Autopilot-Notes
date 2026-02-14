# 自动驾驶周报生成指南

本文档介绍如何为 Autopilot-Notes 项目生成每周的自动驾驶技术周报。

---

## 📁 目录结构

```
ch10_每日前沿/
├── 周报/
│   ├── README.md                    # 本文件
│   ├── weekly-report-generator.md   # 周报生成工作流
│   └── 2026-W07-周报-模板.md         # 周报模板示例
│   ├── 2026-W08-周报.md             # 实际周报文件
│   └── ...
```

---

## 🚀 快速开始

### 生成新周报（推荐流程）

1. **复制模板**
   ```bash
   cd /home/admin/project/Autopilot-Notes/ch10_每日前沿/周报
   cp 2026-W07-周报-模板.md $(date +%Y)-W$(date +%V)-周报.md
   ```

2. **按工作流收集信息**  
   参考 `weekly-report-generator.md` 中的步骤收集本周资讯。

3. **填充内容**  
   编辑新生成的周报文件，替换模板中的占位符内容。

4. **提交更新**
   ```bash
   cd /home/admin/project/Autopilot-Notes
   git add ch10_每日前沿/周报/
   git commit -m "docs: 添加2026年第X周自动驾驶技术周报"
   git push origin main
   ```

---

## 📋 周报内容规范

### 文件命名规则
```
{YYYY}-W{WW}-周报.md

示例:
- 2026-W07-周报.md    (2026年第7周)
- 2026-W08-周报.md    (2026年第8周)
```

### 内容结构要求

1. **热点TOP5**: 按重要性排序，每条约200-300字
2. **技术趋势分析**: 涵盖感知、决策、政策、产业四个维度
3. **公司动态**: 重点跟踪Tesla、Waymo、小鹏、华为等头部企业
4. **下周预告**: 列出值得关注的事件和技术方向

### 信息来源推荐

| 类型 | 推荐来源 |
|------|----------|
| 行业新闻 | 36氪汽车、车东西、智驾网 |
| 技术论文 | arXiv(cs.CV, cs.RO)、CVPR、ICCV |
| 官方动态 | 各公司官方博客、GitHub |
| 政策法规 | 工信部、各地交通委官网 |

---

## 🛠️ 自动化脚本

### 创建新周报

```bash
#!/bin/bash
# create_weekly_report.sh - 创建新的周报文件

YEAR=$(date +%Y)
WEEK=$(date +%V)
FILENAME="${YEAR}-W${WEEK}-周报.md"
TEMPLATE="2026-W07-周报-模板.md"

if [ ! -f "$FILENAME" ]; then
    cp "$TEMPLATE" "$FILENAME"
    echo "✅ 已创建周报: $FILENAME"
    echo "📝 请编辑文件填充内容"
else
    echo "⚠️ 周报已存在: $FILENAME"
fi
```

### 周报提交检查清单

```bash
#!/bin/bash
# pre_commit_check.sh - 提交前检查

echo "🔍 运行周报提交检查..."

# 检查文件是否存在
if [ ! -f "$1" ]; then
    echo "❌ 文件不存在: $1"
    exit 1
fi

# 检查必填字段
checks=(
    "本周热点 TOP5"
    "技术趋势分析"
    "重点关注公司动态"
    "下周值得关注的领域"
)

for check in "${checks[@]}"; do
    if grep -q "$check" "$1"; then
        echo "✅ 检查通过: $check"
    else
        echo "⚠️  缺少内容: $check"
    fi
done

echo "🎉 检查完成!"
```

---

## 📅 发布节奏

- **收集期**: 周一至周六 - 持续收集本周资讯
- **撰写期**: 周日 - 整理撰写周报
- **发布期**: 周日晚间/周一上午 - 提交并推送

---

## 🔗 相关链接

- [周报模板](./2026-W07-周报-模板.md)
- [详细生成流程](./weekly-report-generator.md)
- [项目主页](../../README.md)

---

*Last updated: 2026-02-14*
