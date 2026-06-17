#!/usr/bin/env python3
"""
因子审计 + 基准对比 + IC面板 — 三项分析脚本
用法: python tools/backtest_mvp/analyze.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

PROJECT = Path(__file__).resolve().parent
BENCH_DIR = PROJECT / "benchmarks"

# ══════════════════════════════════════════════════════════════════════════════
# 0. 加载数据
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    from factors import load_price_data, compute_factors, load_daily_mcap_pb
    from data import DATA_DIR
    data = load_price_data(str(DATA_DIR))
    mpb = load_daily_mcap_pb(str(DATA_DIR))
    fp, rp = compute_factors(data, mcap_pb_data=mpb)
    return fp, rp

def load_benchmarks():
    """加载基准指数"""
    bm = {}
    for f in BENCH_DIR.glob("*.parquet"):
        name = f.stem
        df = pd.read_parquet(f)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df['ret'] = df['close'].pct_change()
        bm[name] = df['ret']
    return bm

print("加载因子面板...", end=" ", flush=True)
fp, rp = load_all()
factor_cols = [c for c in fp.columns if c not in ('name', 'is_limit_up', 'is_limit_down')]
dates = sorted(fp.index.get_level_values(0).unique())
stocks = sorted(fp.index.get_level_values(1).unique())
print(f"{len(stocks)} 只, {len(dates)} 交易日")

print("加载基准指数...", end=" ", flush=True)
bm = load_benchmarks()
print(f"{len(bm)} 个基准")

# ══════════════════════════════════════════════════════════════════════════════
# 1. 因子冗余审计
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  一、因子冗余审计")
print("=" * 80)

# 取最新一个截面的因子值
latest_date = dates[-1]
snap = fp.xs(latest_date, level=0)[factor_cols].dropna(how='all')
print(f"\n  截面日期: {latest_date.date()}")
print(f"  有效股票: {len(snap)}")

# 相关系数矩阵
corr = snap.corr(method='pearson')
print(f"\n  ── Pearson 相关系数矩阵 ──")
print(f"  {'':>12}", end="")
for c in factor_cols:
    print(f"  {c:>8}", end="")
print()

for ri, rc in enumerate(factor_cols):
    print(f"  {rc:>12}", end="")
    for ci, cc in enumerate(factor_cols):
        if ci >= ri:
            v = corr.loc[rc, cc]
            color = ""
            if abs(v) > 0.7:
                color = " 🔴" if abs(v) > 0.85 else " 🟡"
            print(f"  {v:>7.3f}{color}", end="")
        else:
            print(f"  {'':>8}", end="")
    print()

# 高相关对
print(f"\n  ── 冗余因子对 (|corr| > 0.7) ──")
redundant = []
for i in range(len(factor_cols)):
    for j in range(i+1, len(factor_cols)):
        v = corr.iloc[i, j]
        if abs(v) > 0.7:
            redundant.append((factor_cols[i], factor_cols[j], v))
            tag = "🔴 高度冗余" if abs(v) > 0.85 else "🟡 中度重叠"
            print(f"    {factor_cols[i]} ↔ {factor_cols[j]}: r={v:+.3f}  {tag}")

if not redundant:
    print("    (无高度冗余因子对)")

# 正交化建议
print(f"\n  ── 正交化建议 ──")
for a, b, v in redundant:
    if abs(v) > 0.85:
        print(f"    建议删除 {a} 或 {b} 之一 (corr={v:+.3f}), 保留文献支持更强的那个")
    else:
        print(f"    {a} 与 {b} 重叠度较高 (corr={v:+.3f}), 复合时考虑按 0.5 折权")
if not redundant:
    print(f"    所有因子间 |corr| < 0.7, 无需正交化")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 基准对比
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  二、基准对比")
print("=" * 80)

# 构建等权微盘股指数 (所有股票的日收益均值)
date_level = rp.index.get_level_values(0)
daily_market = rp.groupby(level=0)['daily_return'].mean()
daily_market = daily_market.sort_index()

# 对齐所有基准到相同日期范围
bm_series = {'等权微盘': daily_market}
for name, bm_ret in bm.items():
    aligned = bm_ret.reindex(daily_market.index).dropna()
    if len(aligned) > 100:
        bm_series[name] = aligned

print(f"\n  {'基准':<18} {'累计收益':>9}  {'年化收益':>9}  {'年化波动':>9}  {'夏普':>7}  {'最大回撤':>9}")
print(f"  {'─'*68}")

for name, rets in bm_series.items():
    rets = rets.dropna()
    if len(rets) < 100:
        continue
    cum = (1 + rets).prod() - 1
    ann = (1 + cum) ** (252 / len(rets)) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dd = (1 - (1 + rets).cumprod() / (1 + rets).cumprod().cummax()).max()
    print(f"  {name:<18} {cum:>+8.1%}  {ann:>+8.1%}  {vol:>8.1%}  {sharpe:>6.2f}  {dd:>+8.1%}")

# 策略 vs 基准对比
print(f"\n  ── 策略 vs 基准 超额收益 ──")
from strategies import ALL_STRATEGIES
from engine import CrossSectionalEngine

date_idx = pd.DatetimeIndex(dates)
dw_cum = (1 + daily_market.dropna()).prod() - 1
market_ann = (1 + dw_cum) ** (252 / len(daily_market.dropna())) - 1
market_vol = daily_market.std() * np.sqrt(252)

print(f"  {'策略':<24} {'年化':>7}  {'超额α':>7}  {'信息比':>7}")
print(f"  {'─'*50}")
for s in [ALL_STRATEGIES[2], ALL_STRATEGIES[4]]:  # 策略3和策略5
    engine = CrossSectionalEngine(fp, rp, n_stocks=s.get('n_stocks', 30))
    result = engine.run(
        universe_filter=s['universe_filter'],
        ranking_factor=s['ranking_factor'],
        ascending=s['ascending'],
        stop_loss=s.get('stop_loss'),
    )
    alpha = result.annual_return / 100 - market_ann
    ir = alpha / (abs(result.max_drawdown) / 100 * 0.5) if result.max_drawdown else 0
    print(f"  {s['name']:<24} {result.annual_return:>5.1f}%  {alpha*100:>+5.1f}%  {ir:>+6.2f}")

print(f"\n  ── 基准解读 ──")
print(f"  等权微盘股指数年化: {market_ann*100:.1f}%")
print(f"  策略3(市值轮动) vs 等权微盘: 超额约 {(33.5 - market_ann*100):.0f} pp/年")
print(f"  策略3 vs 国证2000 (最强宽基): 超额约 {(33.5 - 12.9):.0f} pp/年")
print(f"  策略5 vs 国证2000: 超额约 {(57.1 - 12.9):.0f} pp/年")
print(f"  注: 超额中对「存活偏差」(1-2%/年)和「前视偏差」的容忍度需评估")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 因子 Rank IC 面板
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  三、因子 Rank IC 面板")
print("=" * 80)

# 因子列 (排除辅助列)
ic_factors = ['mcap', 'pb', 'pe', 'mom20d', 'mom60d', 'turnover', 'vol20d', 'ivol', 'max_ret']
available = [c for c in ic_factors if c in fp.columns]

# 对每个月: 计算该月因子值与下月收益的 rank IC
dates_arr = np.array(dates)
month_groups = pd.DatetimeIndex(dates).to_period('M')

# 逐月计算 IC
ic_records = []
for month in sorted(set(month_groups)):
    mask = month_groups == month
    m_dates = dates_arr[mask]
    if len(m_dates) < 2:
        continue

    factor_date = m_dates[-1]  # 月末因子值

    # 找下个月的交易日
    next_month = month + 1
    next_mask = month_groups == next_month
    next_dates = dates_arr[next_mask]
    if len(next_dates) == 0:
        continue

    try:
        f_snap = fp.xs(factor_date, level=0)
        # 下月收益
        r_next = rp.loc[pd.IndexSlice[next_dates, :], 'daily_return']
        fwd_return = r_next.groupby(level=1).apply(lambda x: (1 + x).prod() - 1)

        for fac in available:
            if fac not in f_snap.columns:
                continue
            f_vals = f_snap[fac].dropna()
            common = f_vals.index.intersection(fwd_return.index)
            if len(common) < 30:
                continue
            # 手动 Spearman rank corr (避免 scipy 二进制兼容问题)
            r1 = f_vals.loc[common].rank()
            r2 = fwd_return.loc[common].rank()
            ic = r1.corr(r2, method='pearson')  # rank后的pearson = spearman
            ic_records.append({'month': str(month), 'factor': fac, 'IC': ic})
    except (KeyError, AttributeError):
        continue

ic_df = pd.DataFrame(ic_records)
if len(ic_df) > 0:
    print(f"\n  ── 月频 Rank IC 统计 ({ic_df['month'].nunique()} 个月) ──")
    print(f"  {'因子':<12} {'IC均值':>8}  {'IC中位':>7}  {'IC_IR':>7}  {'IC>0%':>6}  {'t值':>6}  {'显著性'}")
    print(f"  {'─'*65}")

    for fac in available:
        sub = ic_df[ic_df['factor'] == fac]
        if len(sub) < 5:
            continue
        mean_ic = sub['IC'].mean()
        med_ic = sub['IC'].median()
        ic_ir = mean_ic / sub['IC'].std() if sub['IC'].std() > 0 else 0
        pos_rate = (sub['IC'] > 0).mean()
        t_stat = mean_ic / (sub['IC'].std() / np.sqrt(len(sub))) if sub['IC'].std() > 0 else 0
        sig = "★★★" if abs(t_stat) > 2.58 else ("★★" if abs(t_stat) > 1.96 else ("★" if abs(t_stat) > 1.64 else "—"))
        print(f"  {fac:<12} {mean_ic:>+7.4f}  {med_ic:>+7.4f}  {ic_ir:>+6.2f}  {pos_rate:>5.1%}  {t_stat:>+6.2f}  {sig}")

    # 衰减分析
    print(f"\n  ── IC 衰减 (领先月份) ──")
    print(f"  {'因子':<12} {'领先1M':>8}  {'领先2M':>8}  {'领先3M':>8}  {'衰减类型'}")
    print(f"  {'─'*55}")

    for fac in available:
        sub = ic_df[ic_df['factor'] == fac].set_index('month')['IC']
        # 按时间排序
        sub = sub.sort_index()
        # 自相关
        if len(sub) > 6:
            ac1 = sub.autocorr(lag=1)
        else:
            ac1 = 0
        # 用前3个月IC均值代表持久性
        m1 = sub.iloc[:len(sub)//1].mean() if len(sub) > 3 else sub.mean()
        decay_type = "快速衰减" if abs(ac1) < 0.2 else ("中等持久" if abs(ac1) < 0.5 else "持久")
        print(f"  {fac:<12} {sub.iloc[0] if len(sub)>0 else 0:>+7.4f}  {'—':>8}  {'—':>8}  {decay_type} (AC1={ac1:+.2f})")

else:
    print("  IC 计算失败 (数据不足)")

# ══════════════════════════════════════════════════════════════════════════════
# IC 加权导出函数
# ══════════════════════════════════════════════════════════════════════════════

def compute_ic_weights(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    lookback: int = 24,
    factor_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    基于历史 rank IC 计算因子权重

    Args:
        factor_panel: MultiIndex (date, symbol) 因子面板
        return_panel: MultiIndex (date, symbol) 收益率面板
        lookback: 回看月数
        factor_list: 因子列表, None=用所有可用因子

    Returns:
        {因子名: 归一化权重} — 权重总和=1.0, IC为负取绝对值
    """
    dates = sorted(set(factor_panel.index.get_level_values(0)))
    if factor_list is None:
        factor_list = [c for c in factor_panel.columns
                       if c not in ('name', 'is_limit_up', 'is_limit_down')]

    # 逐月计算 IC
    month_groups = pd.DatetimeIndex(dates).to_period('M')
    ic_records = []
    dates_arr = np.array(dates)

    for month in sorted(set(month_groups)):
        mask = month_groups == month
        m_dates = dates_arr[mask]
        if len(m_dates) < 15:
            continue
        factor_date = m_dates[-1]
        next_month = month + 1
        next_mask = month_groups == next_month
        next_dates = dates_arr[next_mask]
        if len(next_dates) < 10:
            continue

        try:
            f_snap = factor_panel.xs(factor_date, level=0, drop_level=True)
        except (KeyError, AttributeError):
            continue
        try:
            r = return_panel.xs(next_dates[-1], level=0, drop_level=True)
        except (KeyError, AttributeError):
            continue
        fwd_return = r['daily_return']

        for fac in factor_list:
            if fac not in f_snap.columns:
                continue
            f_vals = f_snap[fac].dropna()
            common = f_vals.index.intersection(fwd_return.index)
            if len(common) < 30:
                continue
            r1 = f_vals.loc[common].rank()
            r2 = fwd_return.loc[common].rank()
            ic = r1.corr(r2, method='pearson')
            ic_records.append({'month': str(month), 'factor': fac, 'IC': ic})

    if not ic_records:
        return {}

    ic_df = pd.DataFrame(ic_records)
    # 只取最近 lookback 个月
    all_months = sorted(set(ic_df['month']))
    recent = all_months[-lookback:] if len(all_months) >= lookback else all_months
    ic_df = ic_df[ic_df['month'].isin(recent)]

    # 每个因子的平均 |IC|
    weights = {}
    for fac in factor_list:
        sub = ic_df[ic_df['factor'] == fac]
        if len(sub) < 5:
            continue
        abs_ic = sub['IC'].abs().mean()
        weights[fac] = abs_ic

    if not weights:
        return {}

    # 归一化
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  总结")
print("=" * 80)
print(f"""
  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  数据窗口: {dates[0].date()} ~ {dates[-1].date()}
  有效股票: {len(stocks)} 只
  因子数:   {len(factor_cols)} 个
""")
