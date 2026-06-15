#!/bin/bash
# =============================================================================
# 微盘股数据批量采集 Pipeline
# =============================================================================
# 分 4 个阶段, 每阶段有独立限流, 支持断点续传:
#
#   Phase 1: 日线 K 线下载 (westock-data CLI, 限流 0.3s/只)
#   Phase 2: 财务指标拉取 (akshare EM/THS, 限流 1.5s/只)
#   Phase 3: 总股本拉取   (akshare cninfo, 限流 1.0s/只)
#   Phase 4: 逐日 mcap/pb 构建 (纯本地计算, 无限流)
#
# 用法:
#   bash pipeline.sh                     # 全部 4 阶段
#   bash pipeline.sh --phase 1           # 仅日线
#   bash pipeline.sh --phase 2           # 仅财务
#   bash pipeline.sh --max 100           # 最多 100 只
#   bash pipeline.sh --dry-run           # 预览
#   bash pipeline.sh --resume            # 续传 (默认行为)
# =============================================================================

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_STOCKS=500
PHASE=""
DRY_RUN=false

# 解析参数
while [ $# -gt 0 ]; do
    case "$1" in
        --phase)    PHASE="$2"; shift 2 ;;
        --max)      MAX_STOCKS="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --resume)   shift ;;  # 默认行为
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Phase 1: 日线 K 线下载
# =============================================================================
phase1_download_klines() {
    echo ""
    echo "============================================================"
    echo "  Phase 1: 日线 K 线下载 (westock-data CLI, 限流 0.3s)"
    echo "============================================================"

    if $DRY_RUN; then
        echo "  [DRY RUN] 将下载最多 $MAX_STOCKS 只微盘股 ~6 年日线"
        echo "  预计: ${MAX_STOCKS} × 1.3s/只 ≈ $(( MAX_STOCKS * 13 / 10 / 60 )) 分钟"
        return
    fi

    # 先获取列表计算预估
    echo "  获取微盘股列表..."
    TOTAL=$(cd "$DIR" && /Users/hejinyang/miniconda3/bin/python -c "
import sys; sys.path.insert(0, '.')
from data import fetch_microcap_symbols
df = fetch_microcap_symbols()
print(len(df))
" 2>/dev/null || echo "700")

    echo "  候选池: ~$TOTAL 只, 本次目标: $MAX_STOCKS 只"
    echo "  预计下载: $(( MAX_STOCKS * 13 / 10 / 60 )) 分钟 (含限流)"

    START=$(date +%s)
    cd "$DIR" && /Users/hejinyang/miniconda3/bin/python -c "
import sys; sys.path.insert(0, '.')
from data import download_microcap_universe
download_microcap_universe(max_stocks=$MAX_STOCKS, kline_days=1500, skip_existing=True, sleep_between=0.3)
" 2>&1

    ELAPSED=$(($(date +%s) - START))
    CACHE_COUNT=$(ls "$DIR/data_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    echo "  Phase 1 完成: ${CACHE_COUNT} 只, 耗时 ${ELAPSED}s ($((ELAPSED/60))min)"
}

# =============================================================================
# Phase 2: 财务指标拉取
# =============================================================================
phase2_fetch_financials() {
    echo ""
    echo "============================================================"
    echo "  Phase 2: 财务指标拉取 (akshare, 限流 1.5s)"
    echo "============================================================"

    CACHE_COUNT=$(ls "$DIR/data_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    DONE_COUNT=$(ls "$DIR/financials_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    TODO=$((CACHE_COUNT - DONE_COUNT))

    echo "  日线缓存: $CACHE_COUNT 只, 已拉财务: $DONE_COUNT 只, 待拉: $TODO 只"

    if $DRY_RUN; then
        echo "  [DRY RUN] 将拉取 $TODO 只财务指标"
        echo "  预计: ${TODO} × 2.5s/只 ≈ $(( TODO * 25 / 10 / 60 )) 分钟"
        return
    fi

    if [ "$TODO" -eq 0 ]; then
        echo "  ✓ 财务指标已全部完成"
        return
    fi

    START=$(date +%s)
    STOCKS=$(cd "$DIR" && comm -23 <(ls data_cache/*.parquet | xargs -I{} basename {} .parquet | sort) \
                                  <(ls financials_cache/*.parquet 2>/dev/null | xargs -I{} basename {} .parquet | sort))
    TOTAL_TODO=$(echo "$STOCKS" | wc -w | tr -d ' ')

    i=0
    SUCCESS=0
    FAIL=0
    for sym in $STOCKS; do
        i=$((i+1))
        if [ $i -gt 0 ] && [ $(( i % 50 )) -eq 0 ]; then
            ELAPSED=$(($(date +%s) - START))
            ETA=$(( ELAPSED * (TOTAL_TODO - i) / i ))
            printf "\n  [进度] %d/%d (%ds, ETA %ds) 成功:%d 失败:%d\n" $i $TOTAL_TODO $ELAPSED $ETA $SUCCESS $FAIL
        fi

        bash "$DIR/fetch_one.sh" "$sym" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn" | while read -r line; do
            # 只打印关键状态
            :
        done

        if [ -f "$DIR/financials_cache/$sym.parquet" ]; then
            SUCCESS=$((SUCCESS+1))
        else
            FAIL=$((FAIL+1))
        fi
        sleep 1.5
    done

    ELAPSED=$(($(date +%s) - START))
    FIN_COUNT=$(ls "$DIR/financials_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "  Phase 2 完成: 财务指标 ${FIN_COUNT}/${CACHE_COUNT} 只, 耗时 ${ELAPSED}s ($((ELAPSED/60))min)"
}

# =============================================================================
# Phase 3: 总股本拉取
# =============================================================================
phase3_fetch_profiles() {
    echo ""
    echo "============================================================"
    echo "  Phase 3: 总股本拉取 (akshare cninfo, 限流 1.0s)"
    echo "============================================================"

    CACHE_COUNT=$(ls "$DIR/data_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    DONE_COUNT=$(ls "$DIR/profiles_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    TODO=$((CACHE_COUNT - DONE_COUNT))

    echo "  日线缓存: $CACHE_COUNT 只, 已拉股本: $DONE_COUNT 只, 待拉: $TODO 只"

    if $DRY_RUN; then
        echo "  [DRY RUN] 将拉取 $TODO 只总股本"
        echo "  预计: ${TODO} × 1.5s/只 ≈ $(( TODO * 15 / 10 / 60 )) 分钟"
        return
    fi

    if [ "$TODO" -eq 0 ]; then
        echo "  ✓ 总股本已全部完成"
        return
    fi

    START=$(date +%s)
    STOCKS=$(cd "$DIR" && comm -23 <(ls data_cache/*.parquet | xargs -I{} basename {} .parquet | sort) \
                                  <(ls profiles_cache/*.parquet 2>/dev/null | xargs -I{} basename {} .parquet | sort))
    TOTAL_TODO=$(echo "$STOCKS" | wc -w | tr -d ' ')

    i=0
    SUCCESS=0
    for sym in $STOCKS; do
        i=$((i+1))
        if [ $i -gt 0 ] && [ $(( i % 50 )) -eq 0 ]; then
            ELAPSED=$(($(date +%s) - START))
            ETA=$(( ELAPSED * (TOTAL_TODO - i) / i ))
            echo "  [进度] $i/$TOTAL_TODO (${ELAPSED}s, ETA ${ETA}s) 成功:$SUCCESS"
        fi

        # Phase 3 只跑 profile (已缓存 financials 则不会重复拉)
        bash "$DIR/fetch_one.sh" "$sym" 2>&1 | grep -v "RequestsDependencyWarning\|warnings.warn" > /dev/null

        if [ -f "$DIR/profiles_cache/$sym.parquet" ]; then
            SUCCESS=$((SUCCESS+1))
        fi
        sleep 1.0
    done

    ELAPSED=$(($(date +%s) - START))
    PRO_COUNT=$(ls "$DIR/profiles_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "  Phase 3 完成: 总股本 ${PRO_COUNT}/${CACHE_COUNT} 只, 耗时 ${ELAPSED}s ($((ELAPSED/60))min)"
}

# =============================================================================
# Phase 4: 逐日 mcap/pb 构建 (纯本地计算)
# =============================================================================
phase4_build_mcap_pb() {
    echo ""
    echo "============================================================"
    echo "  Phase 4: 逐日 mcap/pb 构建 (本地计算)"
    echo "============================================================"

    CACHE_COUNT=$(ls "$DIR/data_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    DONE_COUNT=$(ls "$DIR/daily_mcap_pb_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    TODO=$((CACHE_COUNT - DONE_COUNT))

    echo "  日线缓存: $CACHE_COUNT 只, 已构建: $DONE_COUNT 只, 待构建: $TODO 只"

    if $DRY_RUN; then
        echo "  [DRY RUN] 将构建 $TODO 只逐日 mcap/pb"
        echo "  预计: < 10 秒 (纯本地)"
        return
    fi

    if [ "$TODO" -eq 0 ]; then
        echo "  ✓ mcap/pb 已全部完成"
        return
    fi

    START=$(date +%s)

    # 找出同时有 financials + profiles + price 但缺 mcap_pb 的股票
    cd "$DIR"
    /Users/hejinyang/miniconda3/bin/python -c "
import pandas as pd
from pathlib import Path

FIN_CACHE = Path('financials_cache')
PRO_CACHE = Path('profiles_cache')
PX_CACHE = Path('data_cache')
MPB_CACHE = Path('daily_mcap_pb_cache')
MPB_CACHE.mkdir(exist_ok=True)

# 找出待构建的
ready = set(f.stem for f in PX_CACHE.glob('*.parquet'))
ready &= set(f.stem for f in FIN_CACHE.glob('*.parquet'))
ready &= set(f.stem for f in PRO_CACHE.glob('*.parquet'))
ready -= set(f.stem for f in MPB_CACHE.glob('*.parquet'))

missing = sorted(ready)
print(f'  待构建: {len(missing)} 只')
if not missing:
    print('  无需构建')
    exit(0)

import numpy as np
import sys
sys.path.insert(0, '.')

for i, sym in enumerate(missing):
    try:
        px = pd.read_parquet(PX_CACHE / f'{sym}.parquet')
        px['date'] = pd.to_datetime(px['date'])
        px = px.sort_values('date')

        shares = pd.read_parquet(PRO_CACHE / f'{sym}.parquet')['total_shares_yi'].iloc[0]
        px['mcap'] = px['close'] * shares

        fin = pd.read_parquet(FIN_CACHE / f'{sym}.parquet')
        fin = fin.dropna(subset=['notice_date']).sort_values('notice_date')

        if 'bps' in fin.columns or 'eps' in fin.columns:
            all_dates = px['date'].values
            bps = np.full(len(all_dates), np.nan)
            epss = np.full(len(all_dates), np.nan)
            for _, r in fin.iterrows():
                nd = r['notice_date']
                if pd.isna(nd): continue
                mask = all_dates >= pd.Timestamp(nd)
                if 'bps' in r and not pd.isna(r['bps']):
                    bps[mask] = r['bps']
                if 'eps' in r and not pd.isna(r['eps']):
                    epss[mask] = r['eps']
            px['bps'] = pd.Series(bps).ffill().values
            px['eps'] = pd.Series(epss).ffill().values
            px['pb'] = px['close'] / px['bps'].replace(0, np.nan)
            px['pe'] = px['close'] / px['eps'].replace(0, np.nan)
            px = px.drop(columns=['bps','eps'], errors='ignore')

        keep = ['date','close','mcap','pb','pe']
        px[[c for c in keep if c in px.columns]].to_parquet(MPB_CACHE / f'{sym}.parquet', index=False)
    except Exception as e:
        if i < 3:
            print(f'    ⚠️ {sym}: {str(e)[:60]}')

    if (i+1) % 100 == 0:
        print(f'  [进度] {i+1}/{len(missing)}')
" 2>&1

    ELAPSED=$(($(date +%s) - START))
    MPB_COUNT=$(ls "$DIR/daily_mcap_pb_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    echo "  Phase 4 完成: mcap/pb ${MPB_COUNT}/${CACHE_COUNT} 只, 耗时 ${ELAPSED}s"
}

# =============================================================================
# 状态报告
# =============================================================================
show_status() {
    echo ""
    echo "============================================================"
    echo "  数据采集状态"
    echo "============================================================"
    PX=$(ls "$DIR/data_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    FIN=$(ls "$DIR/financials_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    PRO=$(ls "$DIR/profiles_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    MPB=$(ls "$DIR/daily_mcap_pb_cache"/*.parquet 2>/dev/null | wc -l | tr -d ' ')

    TOTAL_KB=$(du -sk "$DIR/data_cache" "$DIR/financials_cache" "$DIR/profiles_cache" "$DIR/daily_mcap_pb_cache" 2>/dev/null | awk '{s+=$1} END {print s}')
    echo "  日线:       $PX 只"
    echo "  财务指标:   $FIN 只"
    echo "  总股本:     $PRO 只"
    echo "  逐日mcap/pb: $MPB 只"
    echo "  磁盘占用:   ~$(( TOTAL_KB / 1024 )) MB"
    echo "  ─────────────────────────────────"
    echo "  完整度:     $MPB/$PX ($(( MPB * 100 / (PX > 0 ? PX : 1) ))%)"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   微盘股数据批量采集 Pipeline v1.0                        ║"
echo "║   目标: $MAX_STOCKS 只 | $(date '+%Y-%m-%d %H:%M')               ║"
echo "╚════════════════════════════════════════════════════════════╝"

show_status

if $DRY_RUN; then
    echo "*** DRY RUN 模式 — 不会执行任何下载 ***"
    echo ""
fi

run_all=false
if [ -z "$PHASE" ]; then
    run_all=true
fi

if $run_all || [ "$PHASE" = "1" ]; then
    phase1_download_klines
fi

if $run_all || [ "$PHASE" = "2" ]; then
    phase2_fetch_financials
fi

if $run_all || [ "$PHASE" = "3" ]; then
    phase3_fetch_profiles
fi

if $run_all || [ "$PHASE" = "4" ]; then
    phase4_build_mcap_pb
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Pipeline 完成!                                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
show_status
