#!/bin/bash
# 股东户数数据拉取
# 用法: bash fetch_shareholders.sh [symbol_prefix]  # 如: sh600239
#        bash fetch_shareholders.sh --batch  # 批量拉取所有 data_cache 中的股票
set -e

CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/shareholders_cache"
PYTHON=/Users/hejinyang/miniconda3/bin/python

fetch_one() {
    local sym="$1"  # e.g. sh600239 or sz002630
    local code="${sym:2}"  # strip sh/sz prefix
    local cache_path="$CACHE_DIR/${sym}.parquet"
    
    if [ -f "$cache_path" ]; then
        return 0
    fi
    
    $PYTHON -c "
import akshare as ak
import pandas as pd
from pathlib import Path

try:
    df = ak.stock_zh_a_gdhs(symbol='$code')
    if df is None or len(df) == 0:
        print(f'  $sym: 无数据')
        exit(0)
    
    # Extract: 日期, 股东户数
    keep_cols = [c for c in ['日期', '股东户数'] if c in df.columns]
    if not keep_cols:
        print(f'  $sym: 列名不匹配 ({list(df.columns)[:5]})')
        exit(0)
    
    result = df[keep_cols].copy()
    result.columns = ['date', 'holder_count']
    result['date'] = pd.to_datetime(result['date'], errors='coerce')
    result['holder_count'] = pd.to_numeric(result['holder_count'], errors='coerce')
    result = result.dropna().sort_values('date')
    result['symbol'] = '$sym'
    Path('$CACHE_DIR').mkdir(parents=True, exist_ok=True)
    result.to_parquet('$cache_path', index=False)
    print(f'  $sym: {len(result)} 期, {result[\"date\"].min().date()}~{result[\"date\"].max().date()}')
except Exception as e:
    print(f'  $sym: 失败 ({type(e).__name__})')
" 2>&1 | grep -v 'warnings'
}

if [ "$1" = "--batch" ]; then
    mkdir -p "$CACHE_DIR"
    echo "批量拉取股东户数..."
    total=0; ok=0; fail=0
    for f in "$(dirname "$0")/data_cache"/*.parquet; do
        sym=$(basename "$f" .parquet)
        total=$((total+1))
        fetch_one "$sym" && ok=$((ok+1)) || fail=$((fail+1))
        if [ $((total % 50)) -eq 0 ]; then
            echo "  进度: $total [ok=$ok fail=$fail]"
        fi
        sleep 0.8
    done
    echo "完成: $total 只, ok=$ok, fail=$fail"
else
    mkdir -p "$CACHE_DIR"
    fetch_one "$1"
fi
