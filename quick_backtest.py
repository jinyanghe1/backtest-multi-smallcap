#!/usr/bin/env python3
"""
快速回测脚本 — 端到端下载 + 因子 + 回测 (无需数据缓存)
========================================================
v2: 修复三大审计问题
  - P0: 前视偏差 — 用收盘价动态推算历史mcap/pb (mcap_t = mcap_now * close_t / close_now)
  - P1: quote列索引错位 — 正确解析39列quote输出
  - P2: 滑点每天扣 → 移到调仓日

直接调用 westock-data CLI 下载 K 线, 构建因子面板, 运行全部 11 策略回测。
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

# 添加项目路径
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from engine import CrossSectionalEngine, BacktestResult
from factors import compute_factors
from strategies import ALL_STRATEGIES


def _run_westock(cmd: str, timeout: int = 30) -> str:
    """调用 westock-data CLI, 返回原始输出"""
    result = subprocess.run(
        [str(NODE_BIN), str(WESTOCK_SCRIPT)] + cmd.split(),
        capture_output=True, text=True, timeout=timeout,
        cwd=str(WESTOCK_SCRIPT.parent.parent),
    )
    return result.stdout.strip()


def parse_kline_table(raw: str) -> pd.DataFrame:
    """解析 westock-data 的 Markdown K线表格"""
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
    """解析板块成分股表格, 返回 [code, ...]"""
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
# quote 列索引 (从 westock-data quote 输出解析)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUOTE_COL_INDEX = {
    'code':              0,
    'name':              3,
    'price':             5,    # 最新价
    'prev_close':        6,
    'turnover_rate':    19,    # 换手率 (%)
    'pe_ratio':         22,    # 市盈率 (滚动)
    'pe_fwd':           23,    # 远期市盈率
    'pe_lyr':           24,    # 滚动市盈率
    'pb_ratio':         25,    # 市净率
    'total_market_cap': 29,    # 总市值 (亿)
    'chg_20d':          37,    # 20日涨幅
    'chg_60d':          38,    # 60日涨幅
}


def parse_quote(raw: str) -> dict:
    """
    解析 westock-data 的 quote 输出, 返回 {code: {field: value, ...}}

    列顺序: code, market_type, market_name, name, symbol, price, prev_close,
            open, volume, bid1, bid1_vol, ask1, ask1_vol, time, change,
            change_percent, high, low, amount, turnover_rate, volume_ratio,
            range_pct, pe_ratio, pe_fwd, pe_lyr, pb_ratio, ps_ttm, pcf_ttm,
            dividend_ratio_ttm, total_market_cap, circulating_market_cap,
            total_shares, float_shares, high_52week, low_52week, chg_5d,
            chg_10d, chg_20d, chg_60d, chg_ytd
    """
    lines = raw.split('\n')
    result = {}
    in_table = False
    for line in lines:
        line = line.strip()
        if '|' in line and ('code' in line.lower() or '---' in line):
            in_table = True
            continue
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) < 2:
                continue
            code = parts[0]
            try:
                price = float(parts[QUOTE_COL_INDEX['price']]) if len(parts) > QUOTE_COL_INDEX['price'] else 0
                mcap = float(parts[QUOTE_COL_INDEX['total_market_cap']]) if len(parts) > QUOTE_COL_INDEX['total_market_cap'] else 10.0
                pb = float(parts[QUOTE_COL_INDEX['pb_ratio']]) if len(parts) > QUOTE_COL_INDEX['pb_ratio'] else 2.0
                pe = float(parts[QUOTE_COL_INDEX['pe_ratio']]) if len(parts) > QUOTE_COL_INDEX['pe_ratio'] else 50.0
                turnover = float(parts[QUOTE_COL_INDEX['turnover_rate']]) if len(parts) > QUOTE_COL_INDEX['turnover_rate'] else 2.0
            except (ValueError, IndexError):
                price, mcap, pb, pe, turnover = 0, 10.0, 2.0, 50.0, 2.0
            result[code] = {
                'price': price,
                'mcap_now': mcap,      # 当前市值 (亿)
                'pb_now': pb,           # 当前市净率
                'pe_now': pe,           # 当前市盈率
                'turnover_now': turnover,  # 当前换手率
            }
    return result


def download_and_build(n_stocks: int = 30, kline_days: int = 1000) -> tuple:
    """
    下载微盘股 K 线数据, 构建因子面板和收益率面板

    v2: 修复前视偏差 — 用收盘价动态推算历史市值/市净率

    Returns:
        (factor_panel, return_panel)
    """
    print(f"1. 获取微盘股列表 (前 {n_stocks} 只)...")
    raw = _run_westock("sector pt02GN2282 --limit 100", timeout=30)
    all_codes = parse_sector_table(raw)
    # 过滤北交所 (bj 开头, 流动性差)
    codes = [c for c in all_codes if not c.startswith('bj')][:n_stocks]
    print(f"  获取到 {len(codes)} 只 (过滤北交所)")

    # 获取实时行情 (正确列索引!)
    print(f"2. 获取实时行情 (正确解析quote列索引)...")
    quote_data = {}
    for i in range(0, len(codes), 20):
        batch = ",".join(codes[i:i+20])
        raw_q = _run_westock(f"quote {batch}", timeout=30)
        quote_data.update(parse_quote(raw_q))
        time.sleep(0.3)

    # 打印前3只的解析结果验证
    for code in list(quote_data.keys())[:3]:
        q = quote_data[code]
        print(f"  {code}: mcap={q['mcap_now']:.1f}亿 pb={q['pb_now']:.2f} pe={q['pe_now']:.1f} "
              f"turnover={q['turnover_now']:.1f}% price={q['price']:.2f}")

    # 下载 K 线
    print(f"3. 下载 K 线 (每只 {kline_days} 天)...")
    all_dfs = []
    success = 0
    for i, code in enumerate(codes):
        raw_k = _run_westock(f"kline {code} --period day --limit {kline_days} --fq qfq", timeout=30)
        df = parse_kline_table(raw_k)
        if len(df) > 60:
            df['symbol'] = code
            all_dfs.append(df)
            success += 1

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(codes)} [成功{success}]")
        time.sleep(0.2)

    print(f"  下载完成: {success}/{len(codes)} 只成功")

    if not all_dfs:
        print("⚠️ 无有效数据!")
        return None, None

    # 合并
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)
    print(f"  合并数据: {combined['symbol'].nunique()} 只, {len(combined)} 行")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # P0 修复: 动态推算历史 mcap / pb / pe (消除前视偏差)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 原理: 假设总股本不变, 则 mcap_t = mcap_now * close_t / close_now
    #       同理 pb_t = pb_now * close_t / close_now (假设每股净资产不变)
    #       pe_t 同理 (假设EPS不变, pe_t = pe_now * close_t / close_now)
    # 注意: 这只是近似, 真正的历史mcap需要知道每日总股本+收盘价
    #       但比"用2026年6月的值贴到2022年"好太多
    print("4. 动态推算历史 mcap/pb/pe (消除前视偏差)...")
    combined['mcap'] = np.nan
    combined['pb'] = np.nan
    combined['pe'] = np.nan

    for code, grp in combined.groupby('symbol'):
        q = quote_data.get(code, {})
        mcap_now = q.get('mcap_now', 10.0)
        pb_now = q.get('pb_now', 2.0)
        pe_now = q.get('pe_now', 50.0)
        price_now = q.get('price', 0)

        if price_now <= 0:
            # 无法推算, 用默认值
            combined.loc[grp.index, 'mcap'] = mcap_now
            combined.loc[grp.index, 'pb'] = pb_now
            combined.loc[grp.index, 'pe'] = pe_now
            continue

        # 动态推算: mcap_t = mcap_now * close_t / price_now
        close_t = grp['close'].values
        ratio = close_t / price_now  # 价格变化比率

        combined.loc[grp.index, 'mcap'] = mcap_now * ratio
        combined.loc[grp.index, 'pb'] = pb_now * ratio
        combined.loc[grp.index, 'pe'] = pe_now * ratio

    # 检查动态推算效果
    print(f"  mcap 动态范围: {combined['mcap'].min():.1f} ~ {combined['mcap'].max():.1f} 亿")
    print(f"  pb 动态范围: {combined['pb'].min():.2f} ~ {combined['pb'].max():.2f}")
    print(f"  pe 动态范围: {combined['pe'].min():.1f} ~ {combined['pe'].max():.1f}")

    # 每只股票的 mcap 唯一值数 (验证不再是常量)
    nunique_mcap = combined.groupby('symbol')['mcap'].nunique()
    pct_static = (nunique_mcap == 1).sum() / len(nunique_mcap) * 100
    print(f"  mcap 完全不变的股票: {pct_static:.0f}% {'⚠️' if pct_static > 50 else '✓'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 换手率: 从 volume 数据计算 (不用实时行情的静态值)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 用每日成交量 / 20日均量 作为换手率代理
    # (真正的换手率需要流通股本数据, 这里用相对成交量近似)
    combined = combined.sort_values(['symbol', 'date'])
    combined['turnover'] = combined.groupby('symbol')['volume'].transform(
        lambda x: x / x.rolling(20, min_periods=5).mean().replace(0, 1)
    ).clip(0.01, 100)
    print(f"  turnover 从 volume 计算: 范围 {combined['turnover'].min():.2f} ~ {combined['turnover'].max():.2f}")

    # 计算因子
    print("5. 计算因子 (含PE列)...")
    factor_panel, return_panel = compute_factors(combined, mcap_pb_data=None)

    print(f"  因子面板: {factor_panel.shape}")
    print(f"  收益面板: {return_panel.shape}")
    print(f"  日期范围: {factor_panel.index.get_level_values(0).min().strftime('%Y-%m-%d')} ~ "
          f"{factor_panel.index.get_level_values(0).max().strftime('%Y-%m-%d')}")

    return factor_panel, return_panel


def run_all_backtests(factor_panel, return_panel):
    """运行全部 11 策略回测"""
    print("\n6. 运行回测...")
    print("=" * 110)
    print(f"  {'策略':<30} {'年化收益':>8}  {'夏普':>6}  {'回撤':>7}  {'胜率':>6}  {'换手率':>6}  {'终值倍数':>8}")
    print("  " + "-" * 100)

    results = []
    for s in ALL_STRATEGIES:
        try:
            # 重置 IC 历史 (策略11 用到)
            import strategies
            strategies._ic_history.clear()

            engine = CrossSectionalEngine(
                factor_panel=factor_panel,
                return_panel=return_panel,
                initial_capital=1.0,
                n_stocks=s.get("n_stocks", 30),
                rebalance_freq='M',
                commission=0.00125,  # 万2.5佣金 + 千1印花税 ≈ 0.125%
                slippage=0.002,      # 微盘股滑点 0.2% (仅在调仓日扣除)
            )
            result = engine.run(
                universe_filter=s["universe_filter"],
                ranking_factor=s.get("ranking_factor", "mcap"),
                ascending=s.get("ascending", True),
                stop_loss=s.get("stop_loss"),
                ranking_fn=s.get("ranking_fn"),
            )
            results.append((s["name"], result))
            print(f"  {s['name']:<30} "
                  f"年化 {result.annual_return:>7.2f}%  "
                  f"夏普 {result.sharpe_ratio:>6.2f}  "
                  f"回撤 {result.max_drawdown:>6.2f}%  "
                  f"胜率 {result.win_rate:>5.1f}%  "
                  f"换手 {result.avg_turnover:>5.1f}%  "
                  f"终值 {result.terminal_value:>6.2f}x")
        except Exception as e:
            print(f"  {s['name']:<30} 错误: {str(e)[:60]}")

    # ━━━ 基准对比 ━━━
    print("\n  " + "=" * 100)
    print("  基准: 等权买入持有 (月度再平衡, 无成本)")
    stocks = sorted(factor_panel.index.get_level_values(1).unique())
    dates = sorted(set(factor_panel.index.get_level_values(0))
                  & set(return_panel.index.get_level_values(0)))

    equity = 1.0
    equity_curve = []
    all_dates_idx = pd.DatetimeIndex(dates)
    df_dates = pd.DataFrame({'date': all_dates_idx})
    df_dates['month'] = all_dates_idx.to_period('M')
    rebal_dates = df_dates.groupby('month')['date'].last().values

    for i in range(len(rebal_dates) - 1):
        rd = pd.Timestamp(rebal_dates[i])
        next_rd = pd.Timestamp(rebal_dates[i+1])
        try:
            snap = factor_panel.xs(rd, level=0, drop_level=True)
        except:
            continue
        picked = list(snap.index)
        if len(picked) == 0:
            continue
        w = 1.0 / len(picked)
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
    print(f"  等权基准: 年化 {benchmark_annual:.2f}%, 终值 {total_ret:.3f}x")

    # 汇总
    if results:
        print("\n  " + "=" * 100)
        best_sharpe = max(results, key=lambda x: x[1].sharpe_ratio)
        best_return = max(results, key=lambda x: x[1].annual_return)
        best_calmar = max(results, key=lambda x: x[1].calmar_ratio)
        print(f"  最高夏普: {best_sharpe[0]} → {best_sharpe[1].annual_return}% / Sharpe {best_sharpe[1].sharpe_ratio}")
        print(f"  最高收益: {best_return[0]} → {best_return[1].annual_return}% / 终值 {best_return[1].terminal_value}x")
        print(f"  最高Calmar: {best_calmar[0]} → {best_calmar[1].annual_return}% / 回撤 {best_calmar[1].max_drawdown}%")
        # vs 基准
        for name, result in results:
            alpha = result.annual_return - benchmark_annual
            flag = "✓" if alpha > 0 else "✗"
            print(f"  {name:<30} alpha vs 基准: {alpha:>+7.2f}%/年 {flag}")

    return results


if __name__ == "__main__":
    n = 30
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    factor_panel, return_panel = download_and_build(n_stocks=n, kline_days=1000)
    if factor_panel is not None:
        results = run_all_backtests(factor_panel, return_panel)
