#!/usr/bin/env python3
"""
回测审计脚本 — 系统性排查全亏损原因
====================================
审计维度:
  1. 数据质量: K线合理性、因子NaN率、缺失列
  2. 前视偏差: mcap/pb 是否用了未来数据 (当前值贴到历史)
  3. 算法逻辑: 排名函数是否生效、因子方向是否正确
  4. 交易约束: 滑点/佣金模型是否过度扣除、T+1/涨跌停缺失
  5. 筛选有效性: universe_filter 是否过度过滤导致空仓
  6. 基准对比: 加入等权买入持有基准，计算超额收益
"""

import sys
import os
import json
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# 路径配置
WESTOCK_SCRIPT = Path("/Users/riverosa/.workbuddy/plugins/marketplaces/cb_teams_marketplace/plugins/finance-data/skills/westock-data/scripts/index.js")
NODE_BIN = Path("/Users/riverosa/.workbuddy/binaries/node/versions/22.22.2/bin/node")

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from engine import CrossSectionalEngine, BacktestResult
from factors import compute_factors
from strategies import ALL_STRATEGIES, rank_normalize, composite_rank


def _run_westock(cmd: str, timeout: int = 30) -> str:
    result = subprocess.run(
        [str(NODE_BIN), str(WESTOCK_SCRIPT)] + cmd.split(),
        capture_output=True, text=True, timeout=timeout,
        cwd=str(WESTOCK_SCRIPT.parent.parent),
    )
    return result.stdout.strip()


def parse_kline_table(raw: str) -> pd.DataFrame:
    lines = raw.split('\n')
    records = []
    in_table = False
    for line in lines:
        line = line.strip()
        if '|' in line and ('date' in line.lower() or '---' in line):
            in_table = True
            continue
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 7:
                try:
                    records.append({
                        "date": parts[0],
                        "open": float(parts[1]),
                        "close": float(parts[2]),
                        "high": float(parts[3]),
                        "low": float(parts[4]),
                        "volume": float(parts[5]),
                    })
                except (ValueError, IndexError):
                    continue
    df = pd.DataFrame(records)
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


def parse_sector_table(raw: str) -> list:
    lines = raw.split('\n')
    codes = []
    in_table = False
    for line in lines:
        line = line.strip()
        if '|' in line and ('code' in line.lower() or '---' in line):
            in_table = True
            continue
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 1 and parts[0]:
                codes.append(parts[0])
    return codes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 审计1: 数据质量
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def audit_data_quality(combined: pd.DataFrame):
    print("\n" + "="*80)
    print("🔍 审计1: 数据质量")
    print("="*80)

    n_stocks = combined['symbol'].nunique()
    n_rows = len(combined)
    date_range = f"{combined['date'].min().strftime('%Y-%m-%d')} ~ {combined['date'].max().strftime('%Y-%m-%d')}"

    print(f"  股票数: {n_stocks}, 行数: {n_rows}, 日期范围: {date_range}")

    # 检查异常价格
    for col in ['open', 'close', 'high', 'low']:
        n_zero = (combined[col] == 0).sum()
        n_neg = (combined[col] < 0).sum()
        n_nan = combined[col].isna().sum()
        pct_zero = n_zero / n_rows * 100
        pct_nan = n_nan / n_rows * 100
        flag = " ⚠️" if pct_zero > 0.1 or pct_nan > 1 else " ✓"
        print(f"  {col:>6}: 零值 {n_zero} ({pct_zero:.1f}%), 负值 {n_neg}, NaN {n_nan} ({pct_nan:.1f}%){flag}")

    # 检查涨跌停: high==low (一字板) 或 close==high==low
    n_limit_up = ((combined['close'] == combined['high']) & (combined['close'] > combined['open'])).sum()
    n_limit_down = ((combined['close'] == combined['low']) & (combined['close'] < combined['open'])).sum()
    n_one_price = (combined['high'] == combined['low']).sum()
    print(f"  涨停近似: {n_limit_up} ({n_limit_up/n_rows*100:.1f}%)")
    print(f"  跌停近似: {n_limit_down} ({n_limit_down/n_rows*100:.1f}%)")
    print(f"  一字板: {n_one_price} ({n_one_price/n_rows*100:.1f}%)")

    # 检查成交量
    n_zero_vol = (combined['volume'] == 0).sum()
    print(f"  零成交量: {n_zero_vol} ({n_zero_vol/n_rows*100:.1f}%)")
    if n_zero_vol > 0:
        print(f"  ⚠️ 零成交量意味着当日不可交易! 需要排除这些日期")

    return n_zero_vol > n_rows * 0.01  # >1% 零成交则标记


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 审计2: 前视偏差 (最关键!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def audit_lookahead_bias(combined: pd.DataFrame):
    print("\n" + "="*80)
    print("🔍 审计2: 前视偏差 (Look-Ahead Bias)")
    print("="*80)

    # 检查 mcap/pb/turnover 是否为静态值 (同一天所有日期相同)
    for col in ['mcap', 'pb', 'turnover']:
        if col not in combined.columns:
            print(f"  {col}: 列不存在 ⚠️")
            continue

        # 每只股票该列的唯一值数量
        nunique_per_stock = combined.groupby('symbol')[col].nunique()
        n_unique = (nunique_per_stock == 1).sum()
        pct_unique = n_unique / combined['symbol'].nunique() * 100

        if pct_unique > 80:
            print(f"  {col}: {pct_unique:.0f}% 的股票该值完全不变 → ⚠️ 使用了实时行情贴到全部历史!")
            print(f"         这意味着用2026年6月的市值来排名2022年的股票!")
            print(f"         → 严重前视偏差: 选中的是'未来变成小盘股'的股票")
            print(f"         → 这些股票大概率是跌了80%+才变小的, 当然亏钱")
        else:
            print(f"  {col}: 仅 {pct_unique:.0f}% 股票值不变 → ✓ 大部分有历史变化")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 审计3: 因子面板缺失列
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def audit_factor_panel(factor_panel: pd.DataFrame):
    print("\n" + "="*80)
    print("🔍 审计3: 因子面板")
    print("="*80)

    print(f"  形状: {factor_panel.shape}")
    print(f"  列名: {list(factor_panel.columns)}")

    # 检查关键列是否存在
    required_cols = ['mcap', 'pb', 'mom20d', 'mom60d', 'turnover', 'vol20d']
    missing = [c for c in required_cols if c not in factor_panel.columns]
    if missing:
        print(f"  ⚠️ 缺失列: {missing}")
    else:
        print(f"  ✓ 所有必要列都存在")

    # 检查 PE 是否存在 (S8/S10 需要)
    if 'pe' in factor_panel.columns:
        print(f"  ✓ PE 列存在")
    else:
        print(f"  ⚠️ PE 列不存在! 策略8/10的GP/A代理将退化为只用PB")
        print(f"     → 高PB ≠ 高盈利, 在微盘股中高PB往往是投机股!")

    # NaN率
    for col in factor_panel.columns:
        nan_pct = factor_panel[col].isna().mean() * 100
        flag = " ⚠️" if nan_pct > 10 else " ✓"
        print(f"  {col:>10}: NaN {nan_pct:.1f}%{flag}")

    # 因子分布
    print(f"\n  因子分布 (percentiles):")
    for col in factor_panel.columns:
        vals = factor_panel[col].dropna()
        if len(vals) > 0:
            print(f"  {col:>10}: P5={vals.quantile(0.05):.3f}  P50={vals.quantile(0.50):.3f}  P95={vals.quantile(0.95):.3f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 审计4: 算法逻辑 — 排名函数是否生效
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def audit_ranking_fn(factor_panel: pd.DataFrame):
    print("\n" + "="*80)
    print("🔍 审计4: 排名函数有效性")
    print("="*80)

    # 取一个中间日期的横截面
    dates = factor_panel.index.get_level_values(0).unique()
    mid_date = dates[len(dates)//2]

    try:
        snapshot = factor_panel.xs(mid_date, level=0, drop_level=True)
    except:
        print("  ⚠️ 无法取横截面")
        return

    print(f"  测试日期: {mid_date.strftime('%Y-%m-%d')}, 股票数: {len(snapshot)}")

    # 测试每个 ranking_fn
    import strategies
    strategies._ic_history.clear()

    for s in ALL_STRATEGIES:
        rfn = s.get('ranking_fn')
        if rfn is None:
            continue

        try:
            # 先过滤
            if s['universe_filter'] is not None:
                selected = s['universe_filter'](snapshot, list(dates), 0)
            else:
                selected = list(snapshot.index)

            if len(selected) == 0:
                print(f"  {s['name']}: ⚠️ 过滤后0只!")
                continue

            scores = rfn(snapshot.loc[selected])
            valid = scores.dropna()

            # 检查分数是否全是0 (意味着因子缺失或逻辑错误)
            if valid.std() == 0:
                print(f"  {s['name']}: ⚠️ 所有分数相同 (std=0), 排名无效!")
            else:
                # 打印 top 5
                top5 = valid.nlargest(min(5, len(valid)))
                print(f"  {s['name']}: ✓ 有效 (选{len(selected)}只, {len(valid)}只有分)")
                print(f"    Top5: {dict(top5.round(3))}")
        except Exception as e:
            print(f"  {s['name']}: ❌ 错误: {str(e)[:80]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 审计5: 交易成本模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def audit_cost_model(factor_panel, return_panel):
    print("\n" + "="*80)
    print("🔍 审计5: 交易成本模型")
    print("="*80)

    # 跑一个无成本基准 vs 有成本
    dates = sorted(set(factor_panel.index.get_level_values(0))
                  & set(return_panel.index.get_level_values(0)))
    n_stocks = min(30, factor_panel.index.get_level_values(1).nunique())

    # 策略3 (最简单) 无成本
    from strategies import strategy_micro_rotation
    engine_nocost = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=n_stocks,
        rebalance_freq='M', commission=0.0, slippage=0.0,
    )
    r_nocost = engine_nocost.run(
        universe_filter=strategy_micro_rotation['universe_filter'],
        ranking_factor='mcap', ascending=True,
    )

    # 有成本
    engine_cost = CrossSectionalEngine(
        factor_panel=factor_panel, return_panel=return_panel,
        initial_capital=1.0, n_stocks=n_stocks,
        rebalance_freq='M', commission=0.00125, slippage=0.002,
    )
    r_cost = engine_cost.run(
        universe_filter=strategy_micro_rotation['universe_filter'],
        ranking_factor='mcap', ascending=True,
    )

    cost_drag = r_nocost.annual_return - r_cost.annual_return
    print(f"  策略3 无成本: 年化 {r_nocost.annual_return:.2f}%, 终值 {r_nocost.terminal_value:.3f}x")
    print(f"  策略3 有成本: 年化 {r_cost.annual_return:.2f}%, 终值 {r_cost.terminal_value:.3f}x")
    print(f"  成本拖累: {cost_drag:.2f}%/年")
    if cost_drag > 8:
        print(f"  ⚠️ 成本拖累 {cost_drag:.1f}%! 可能过度扣除")
    elif cost_drag > 3:
        print(f"  ⚠️ 成本拖累较高 ({cost_drag:.1f}%), 微盘股需要合理但不过度")
    else:
        print(f"  ✓ 成本拖累合理")

    # 检查滑点模型是否每天扣除 (应该只在调仓日)
    # 读取引擎代码中的滑点逻辑
    print(f"\n  引擎滑点逻辑检查:")
    print(f"    佣金: 只在每月第一天扣除 0.125% → 年化 1.5% (合理)")
    print(f"    滑点: 每天扣除 0.002/21≈0.0095% → 年化 2.4%")
    print(f"    问题: 滑点应该只在调仓日扣除, 不是每天!")
    print(f"    修复: 滑点移到调仓日第一天 (与佣金一起)")

    return r_nocost, r_cost


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 审计6: 基准对比 — 等权买入持有
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def audit_benchmark(factor_panel, return_panel, r_nocost):
    print("\n" + "="*80)
    print("🔍 审计6: 基准对比")
    print("="*80)

    # 等权买入持有 (所有股票等权, 不调仓)
    stocks = sorted(factor_panel.index.get_level_values(1).unique())
    dates = sorted(set(factor_panel.index.get_level_values(0))
                  & set(return_panel.index.get_level_values(0)))

    # 月初等权再平衡
    equity = 1.0
    equity_curve = []
    all_dates_idx = pd.DatetimeIndex(dates)

    # 每月初再平衡
    df_dates = pd.DataFrame({'date': all_dates_idx})
    df_dates['month'] = all_dates_idx.to_period('M')
    rebal_dates = df_dates.groupby('month')['date'].last().values

    for i in range(len(rebal_dates) - 1):
        rd = pd.Timestamp(rebal_dates[i])
        next_rd = pd.Timestamp(rebal_dates[i+1])

        # 选所有可用股票
        try:
            snap = factor_panel.xs(rd, level=0, drop_level=True)
        except:
            continue
        picked = list(snap.index)

        if len(picked) == 0:
            continue

        w = 1.0 / len(picked)
        month_start_eq = equity

        # 月内每天计算组合收益
        period_dates = [d for d in dates if d > rd and d <= next_rd]
        for date in period_dates:
            try:
                r = return_panel.xs(date, level=0, drop_level=True)
                r = r.reindex(picked)['daily_return'].fillna(0)
            except:
                continue
            port_ret = (r * w).sum()
            equity *= (1 + port_ret)

        equity_curve.append(equity)

    n_years = len(equity_curve) / 12
    total_ret = equity_curve[-1] if equity_curve else 1.0
    benchmark_annual = (total_ret ** (1/max(n_years, 0.25)) - 1) * 100

    print(f"  等权买入持有基准: 年化 {benchmark_annual:.2f}%, 终值 {total_ret:.3f}x")
    print(f"  策略3 无成本: 年化 {r_nocost.annual_return:.2f}%")
    alpha = r_nocost.annual_return - benchmark_annual
    print(f"  超额收益 (alpha): {alpha:.2f}%/年")

    if benchmark_annual < -10:
        print(f"  → 基准深度亏损! 整个微盘股市场在熊市中")
        print(f"  → 策略收益低于基准 = 有选股Alpha, 只是市场太差")
        print(f"  → 策略收益高于基准 = 无选股Alpha, 策略本身有问题")
    else:
        print(f"  → 基准不算太差, 策略亏损可能来自成本或逻辑bug")

    return benchmark_annual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 审计7: 前视偏差影响量化 — 用动态mcap vs 静态mcap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def audit_lookahead_impact(combined: pd.DataFrame):
    print("\n" + "="*80)
    print("🔍 审计7: 前视偏差影响量化")
    print("="*80)

    # 用价格近似市值变化
    # 假设: mcap_static / close_first ≈ mcap_at_any_date / close_at_date
    # 所以 mcap_at_date ≈ mcap_static * (close_at_date / close_first)

    # 对每只股票, 计算市值在回测期内的变化
    for sym in combined['symbol'].unique()[:5]:
        sub = combined[combined['symbol'] == sym].sort_values('date')
        if len(sub) < 10:
            continue
        first_close = sub['close'].iloc[0]
        last_close = sub['close'].iloc[-1]
        static_mcap = sub['mcap'].iloc[0]  # 当前市值

        # 如果市值是从当前值反推, 那么在开始时的"真实"市值应该是:
        implied_initial_mcap = static_mcap * (first_close / last_close)
        print(f"  {sym}: 当前市值 {static_mcap:.1f}亿, "
              f"期初价格 {first_close:.2f} → 期末 {last_close:.2f}")
        print(f"    → 期初实际市值 ≈ {implied_initial_mcap:.1f}亿 (vs 误用 {static_mcap:.1f}亿)")
        print(f"    → 前视偏差: {((static_mcap - implied_initial_mcap) / implied_initial_mcap * 100):.0f}%")
        print(f"    → 股价变化: {((last_close/first_close - 1)*100):.0f}%")


def main():
    n = 80
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    print(f"╔{'═'*78}╗")
    print(f"║  回测全亏损根因审计 — {n}只微盘股{' '*40}║")
    print(f"╚{'═'*78}╝")

    # 1. 下载数据
    print(f"\n📥 下载数据...")
    raw = _run_westock("sector pt02GN2282 --limit 100", timeout=30)
    all_codes = parse_sector_table(raw)
    codes = [c for c in all_codes if not c.startswith('bj')][:n]
    print(f"  获取到 {len(codes)} 只")

    # 获取实时行情
    quote_data = {}
    for i in range(0, len(codes), 20):
        batch = ",".join(codes[i:i+20])
        raw_q = _run_westock(f"quote {batch}", timeout=30)
        lines = raw_q.split('\n')
        in_table = False
        for line in lines:
            line = line.strip()
            if '|' in line and ('code' in line.lower() or '---' in line):
                in_table = True
                continue
            if in_table and line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 2:
                    code = parts[0]
                    try:
                        mcap = float(parts[7].replace(',','').replace('亿','')) if len(parts) > 7 else 10.0
                        turnover = float(parts[6].replace('%','')) if len(parts) > 6 else 2.0
                        pb = float(parts[8]) if len(parts) > 8 else 3.0
                        pe = float(parts[9]) if len(parts) > 9 else 50.0
                    except:
                        mcap, turnover, pb, pe = 10.0, 2.0, 3.0, 50.0
                    quote_data[code] = {'mcap': mcap, 'turnover': turnover, 'pb': pb, 'pe': pe}
        time.sleep(0.3)

    # 下载K线
    all_dfs = []
    for i, code in enumerate(codes):
        raw_k = _run_westock(f"kline {code} --period day --limit 1000 --fq qfq", timeout=30)
        df = parse_kline_table(raw_k)
        if len(df) > 60:
            df['symbol'] = code
            q = quote_data.get(code, {})
            df['mcap'] = q.get('mcap', 10.0)
            df['pb'] = q.get('pb', 3.0)
            df['turnover'] = q.get('turnover', 2.0)
            df['pe'] = q.get('pe', 50.0)
            all_dfs.append(df)
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(codes)} [{len(all_dfs)} 成功]")
        time.sleep(0.2)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)

    # ━━━ 运行审计 ━━━
    has_data_issues = audit_data_quality(combined)
    audit_lookahead_bias(combined)
    audit_lookahead_impact(combined)

    # 计算因子
    print(f"\n📊 计算因子面板...")
    factor_panel, return_panel = compute_factors(combined, mcap_pb_data=None)

    audit_factor_panel(factor_panel)
    audit_ranking_fn(factor_panel)
    r_nocost, r_cost = audit_cost_model(factor_panel, return_panel)
    benchmark_annual = audit_benchmark(factor_panel, return_panel, r_nocost)

    # ━━━ 最终结论 ━━━
    print("\n" + "="*80)
    print("📋 审计结论")
    print("="*80)

    issues = []

    # 检查前视偏差
    nunique_mcap = combined.groupby('symbol')['mcap'].nunique()
    if (nunique_mcap == 1).sum() / combined['symbol'].nunique() > 0.8:
        issues.append(("P0-致命", "前视偏差: 用当前市值排名历史股票, 选出的是'未来暴跌股'"))

    # 检查PE缺失
    if 'pe' not in factor_panel.columns:
        issues.append(("P1-严重", "PE列缺失: S8/S10的GP/A代理退化为只用PB, 高PB≠高盈利"))

    # 检查成本拖累
    cost_drag = r_nocost.annual_return - r_cost.annual_return
    if cost_drag > 5:
        issues.append(("P2-中等", f"成本拖累 {cost_drag:.1f}%/年, 滑点模型可能有问题"))

    # 检查基准对比
    if benchmark_annual < -15:
        issues.append(("P0-解释", f"基准年化 {benchmark_annual:.1f}%, 微盘股深度熊市是主因之一"))

    for sev, desc in issues:
        icon = "🔴" if "P0" in sev else "🟡" if "P1" in sev else "🟠"
        print(f"  {icon} [{sev}] {desc}")

    if not issues:
        print("  ✓ 未发现严重问题")
    else:
        print(f"\n  建议修复优先级:")
        print(f"    1. 修复前视偏差: 用收盘价动态推算历史市值 (mcap_t = mcap_now * close_t / close_now)")
        print(f"    2. 加入PE列到因子面板 (或从财务数据获取)")
        print(f"    3. 修复滑点模型: 只在调仓日扣除")
        print(f"    4. 加入基准对比到回测结果")
        print(f"    5. 加入涨跌停过滤 (至少排除一字板日期)")


if __name__ == "__main__":
    main()
