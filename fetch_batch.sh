#!/bin/bash
# 批量轮询财务数据 (每只股票独立进程, 避免沙箱子进程阻隔)
# Usage: bash fetch_batch.sh [--retry-failed]

DIR="$(cd "$(dirname "$0")" && pwd)"
DELAY=0.5

# 获取股票列表
if [ "$1" = "--retry-failed" ]; then
    SYMBOLS=""
    for f in "$DIR/data_cache"/*.parquet; do
        sym=$(basename "$f" .parquet)
        if [ ! -f "$DIR/financials_cache/$sym.parquet" ] || [ ! -f "$DIR/profiles_cache/$sym.parquet" ]; then
            SYMBOLS="$SYMBOLS $sym"
        fi
    done
    SYMBOLS=$(echo $SYMBOLS | xargs)
    echo "重试模式: $(echo $SYMBOLS | wc -w) 只待处理"
else
    SYMBOLS=$(ls "$DIR/data_cache"/*.parquet 2>/dev/null | xargs -I{} basename {} .parquet | sort)
fi

if [ -z "$SYMBOLS" ]; then
    echo "没有需要处理的股票。"
    exit 0
fi

TOTAL=$(echo "$SYMBOLS" | wc -w)
echo "开始轮询 $TOTAL 只股票..."
echo ""

i=0
SUCCESS=0
FAIL=0
for sym in $SYMBOLS; do
    i=$((i+1))
    printf "[%d/%d] " $i $TOTAL
    bash "$DIR/fetch_one.sh" "$sym" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn" || echo "$sym: ERROR"
    # 检查成功
    if [ -f "$DIR/financials_cache/$sym.parquet" ] && [ -f "$DIR/profiles_cache/$sym.parquet" ]; then
        SUCCESS=$((SUCCESS+1))
    else
        FAIL=$((FAIL+1))
    fi
    sleep "$DELAY"
done

echo ""
echo "============================================================"
echo "  完成: $SUCCESS 成功, $FAIL 失败 (共 $TOTAL)"
echo "============================================================"

FIN_COUNT=$(ls "$DIR/financials_cache"/*.parquet 2>/dev/null | wc -l)
PRO_COUNT=$(ls "$DIR/profiles_cache"/*.parquet 2>/dev/null | wc -l)
MPB_COUNT=$(ls "$DIR/daily_mcap_pb_cache"/*.parquet 2>/dev/null | wc -l)
echo "  财务指标: $FIN_COUNT | 总股本: $PRO_COUNT | 逐日mcap/pb: $MPB_COUNT"
if [ $FAIL -gt 0 ]; then
    echo ""
    echo "  重试命令: bash $DIR/fetch_batch.sh --retry-failed"
fi
