#!/usr/bin/env bash
# =============================================================================
# 无人驾驶技术日报自动生成脚本
# 用途: 调用 LLM API 生成当日日报，归档 7 天前的日报，刷新索引
# 运行环境: GitHub Actions (ubuntu-latest) 或本地 (macOS/Linux)
#
# 环境变量:
#   LLM_API_KEY   (必填) LLM API Key（OpenAI 兼容格式）
#   LLM_BASE_URL  (可选) API 地址，默认 https://api.deepseek.com
#   LLM_MODEL     (可选) 模型名，默认 deepseek-chat
#   TODAY         (可选) 指定生成日期 YYYY-MM-DD，默认北京时间今天
#   MOCK          (可选) 1 = 不调用 API，写入示例内容（用于测试）
#   DRY_RUN       (可选) 1 = 只打印将执行的动作，不写文件不调 API
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DAILY_DIR="$REPO_DIR/ch10_每日前沿"
ARCHIVE_DIR="$DAILY_DIR/archive"
KEEP_DAYS=7                     # 日报保留天数（主目录）
REF_DATE="2026-02-23"           # 参考日期（当日方向=厂商方案，用于方向轮换）
DIRECTIONS=("厂商方案" "实战内容" "前沿跟踪" "产品篇")

# ---------- 跨平台日期工具 ----------
date_epoch() { # date_epoch YYYY-MM-DD -> 秒
  if [[ "$(uname)" == "Darwin" ]]; then
    date -j -f "%Y-%m-%d" "$1" +%s
  else
    date -d "$1" +%s
  fi
}

# ---------- 计算今日与方向 ----------
TODAY="${TODAY:-$(TZ='Asia/Shanghai' date +%Y-%m-%d)}"
FILE="$DAILY_DIR/$TODAY.md"
DIFF_DAYS=$(( ( $(date_epoch "$TODAY") - $(date_epoch "$REF_DATE") ) / 86400 ))
DIRECTION="${DIRECTIONS[$(( DIFF_DAYS % 4 ))]}"

echo "📅 日报日期: $TODAY | 方向: $DIRECTION"
echo "📂 仓库目录: $REPO_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY-RUN] 仅演练，不写文件不调 API"
fi

# 子命令: generate(默认) | refresh(只刷新索引)
CMD="${1:-generate}"

# ---------- 0. 刷新索引函数 ----------
refresh_index() {
  echo "🔄 刷新日报索引..."
  python3 - "$DAILY_DIR" <<'PYEOF'
import glob, os, re, sys
from collections import OrderedDict

daily_dir = sys.argv[1]
archive_dir = os.path.join(daily_dir, "archive")
repo_root = os.path.dirname(daily_dir)

def list_dailies(d):
    return [f for f in glob.glob(os.path.join(d, "20??-??-??.md"))
            if re.search(r"/20\d{2}-\d{2}-\d{2}\.md$", f)]

def date_of(f):
    return os.path.basename(f)[:10]

def keywords(f):
    for line in open(f, encoding="utf-8"):
        m = re.search(r"关键词[：:]\s*(.+)", line)
        if m:
            return m.group(1).strip()
    for line in open(f, encoding="utf-8"):
        m = re.match(r"###\s+\d+[\.、]\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return ""

all_files = list_dailies(daily_dir) + list_dailies(archive_dir)
all_dates = sorted(date_of(f) for f in all_files)
recent = sorted(list_dailies(daily_dir), key=date_of, reverse=True)[:7]

def build_table(prefix, heading):
    months = OrderedDict()
    for f in recent:
        d = date_of(f)
        months.setdefault(d[:7], []).append((d, f))
    lines = []
    if heading:
        lines += ["## 📅 最近日报", ""]
    for month, items in months.items():
        lines.append(f"### {month[:4]}年{int(month[5:7])}月")
        lines.append("")
        lines.append("| 日期 | 标题 | 关键词 |")
        lines.append("|------|------|--------|")
        for d, f in items:
            lines.append(f"| {d[5:]} | [{os.path.basename(f)}]({prefix}{os.path.basename(f)}) | {keywords(f)} |")
        lines.append("")
    return "\n".join(lines).rstrip()

targets = [
    {"readme": os.path.join(daily_dir, "README.md"),
     "begin": "<!-- BEGIN_DAILY_INDEX -->", "end": "<!-- END_DAILY_INDEX -->",
     "table": build_table("./", True) + "\n\n> 📦 7 天前的日报已自动归档至 [archive](./archive/)",
     "count": True, "time": True},
    {"readme": os.path.join(repo_root, "README.md"),
     "begin": "<!-- BEGIN_HOME_INDEX -->", "end": "<!-- END_HOME_INDEX -->",
     "table": build_table("./ch10_每日前沿/", False) + "\n\n> 📦 7 天前的日报已自动归档至 [archive](./ch10_每日前沿/archive/)",
     "count": False, "time": False},
]

for t in targets:
    with open(t["readme"], encoding="utf-8") as fh:
        content = fh.read()
    b = content.find(t["begin"])
    e = content.find(t["end"])
    if b != -1 and e != -1:
        content = content[:b] + t["begin"] + "\n" + t["table"] + "\n" + t["end"] + content[e + len(t["end"]):]
    if t["count"]:
        content = re.sub(r"- \*\*总日报数\*\*：\d+篇", f"- **总日报数**：{len(all_files)}篇", content)
    if t["time"] and all_dates:
        content = re.sub(r"- \*\*覆盖时间\*\*：[\d\-]+ 至今", f"- **覆盖时间**：{all_dates[0]} 至今", content)
    with open(t["readme"], "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"📝 索引已更新: {os.path.relpath(t['readme'], repo_root)}")
PYEOF
}

if [[ "$CMD" == "refresh" ]]; then
  refresh_index
  echo "🎉 完成"
  exit 0
fi

# ---------- 1. 归档 7 天前的日报 ----------
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "$ARCHIVE_DIR"
  for f in "$DAILY_DIR"/20??-??-??.md; do
    [[ -e "$f" ]] || continue
    base="$(basename "$f")"
    d="${base%.md}"
    age=$(( ( $(date_epoch "$TODAY") - $(date_epoch "$d") ) / 86400 ))
    if (( age > KEEP_DAYS )); then
      echo "📦 归档: $base (已 $age 天)"
      git mv "$f" "$ARCHIVE_DIR/" 2>/dev/null || mv "$f" "$ARCHIVE_DIR/"
    fi
  done
fi

# ---------- 2. 今日已存在则退出 ----------
if [[ -f "$FILE" ]]; then
  echo "✅ 今日日报已存在: ${FILE}，跳过生成"
  exit 0
fi

# ---------- 3. 读取昨日日报用于去重 ----------
YESTERDAY_SUMMARY=""
YESTERDAY_FILE="$(ls "$DAILY_DIR"/20??-??-??.md 2>/dev/null | sort | tail -1 || true)"
if [[ -n "$YESTERDAY_FILE" ]]; then
  YESTERDAY_SUMMARY="$(head -c 3000 "$YESTERDAY_FILE")"
  echo "🗞️ 参考昨日日报去重: $(basename "$YESTERDAY_FILE")"
fi

# ---------- 4. 生成内容 ----------
if [[ "${MOCK:-0}" == "1" ]]; then
  echo "🧪 MOCK 模式：写入示例内容（不调 API）"
  CONTENT=$(cat <<'MOCKEOF'
### 1. 示例条目A
- **摘要**：这是 MOCK 模式生成的示例内容，用于本地测试流程。
- **时间**：__TODAY__
- **来源**：测试机构

### 2. 示例条目B
- **摘要**：第二条示例，验证多条目与关键词解析。
- **时间**：__TODAY__
- **来源**：测试机构
MOCKEOF
)
  CONTENT="${CONTENT//__TODAY__/$TODAY}"
else
  if [[ -z "${LLM_API_KEY:-}" ]]; then
    echo "❌ 未设置 LLM_API_KEY（请配置 GitHub Actions Secret）" >&2
    exit 1
  fi
  echo "🤖 调用 LLM 生成日报 (model=$LLM_MODEL)..."

  # 构造 payload（用 python3 json.dumps 保证转义安全）
  python3 - "$DIRECTION" "$TODAY" "$YESTERDAY_SUMMARY" > /tmp/daily_payload.json <<'PYEOF'
import json, os, sys
import datetime
direction, today, yesterday = sys.argv[1], sys.argv[2], sys.argv[3]
wd = ["一", "二", "三", "四", "五", "六", "日"][datetime.date.fromisoformat(today).weekday()]
system = (
    "你是《无人驾驶技术日报》的首席编辑，熟悉全球自动驾驶行业动态"
    "（特斯拉、Waymo、华为、小鹏、蔚来、理想、地平线、英伟达、百度Apollo、"
    "小马智行、元戎启行、Momenta、Cruise、Zoox 等）及前沿技术"
    "（端到端、VLA、世界模型、Occupancy Network、车路云一体化、Robotaxi 商业化等）。"
)
user = f"""请生成 {today}（星期{wd}）的《无人驾驶技术日报》，今日主题方向：{direction}。

要求：
1. 内容基于你掌握的真实行业信息，共 10 条核心价值信息，围绕主题方向组织为 3~4 个板块，每个板块 3~4 条；每条格式：
   ### N. 标题
   - **摘要**：2~3 句话
   - **时间**：{today}
   - **来源**：机构/公司名
2. 开头必须包含：
   # 无人驾驶技术日报 - {today}
   > 内容来源：AI 智能精选 | 10条核心价值信息 | 今日方向：{direction}
   > 关键词：关键词1、关键词2、……（8~15 个，用于索引）
3. 与昨日内容去重（不要重复收录相同事件），昨日日报要点如下：
   {yesterday}
4. 如果过去 24 小时确实没有值得收录的新信息，只输出一行：SKIP
5. 只输出 markdown 正文，不要任何额外说明。"""
print(json.dumps({
    "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
    "temperature": 0.7,
    "max_tokens": 3000,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
}, ensure_ascii=False))
PYEOF

  RESP="$(curl -sS -f --max-time 120 -X POST "${LLM_BASE_URL:-https://api.deepseek.com}/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${LLM_API_KEY}" \
    --data @/tmp/daily_payload.json)" || { echo "❌ LLM API 调用失败" >&2; exit 1; }

  CONTENT="$(printf '%s' "$RESP" | jq -r '.choices[0].message.content // empty' | python3 -c '
import sys, re
s = sys.stdin.read().strip()
m = re.match(r"^```[a-zA-Z]*\s*\n", s)
if m: s = s[m.end():]
if s.endswith("```"): s = s[:-3]
# 去除 LLM 可能重复输出的头部（脚本会统一写入头部）
lines = s.split("\n")
while lines and (lines[0].startswith("# 无人驾驶技术日报") or lines[0].startswith("> 内容来源")):
    lines.pop(0)
s = "\n".join(lines).strip()
print(s)
')"
  if [[ -z "$CONTENT" ]]; then
    echo "❌ LLM 返回内容为空" >&2
    exit 1
  fi
fi

# ---------- 5. SKIP 检测（周末/无重大新闻自动跳过） ----------
if [[ "${MOCK:-0}" != "1" ]] && [[ "$CONTENT" =~ ^SKIP ]]; then
  echo "⏭️ 今日无重大新闻 (SKIP)，不生成日报"
  exit 0
fi

# ---------- 6. 写入日报文件 ----------
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY-RUN] 将写入: $FILE"
else
  cat > "$FILE" <<EOF
# 无人驾驶技术日报 - $TODAY

> 内容来源：AI 智能精选 | 10条核心价值信息 | 今日方向：$DIRECTION

$CONTENT

---

*本日报由 AI 自动生成，每日更新 | 方向轮换：产品篇→厂商方案→实战内容→前沿跟踪*
EOF
  echo "✅ 已生成: $FILE ($(wc -c < "$FILE") 字节)"
fi

# ---------- 7. 刷新索引 (ch10_每日前沿/README.md) ----------
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  refresh_index
fi

echo "🎉 完成"
