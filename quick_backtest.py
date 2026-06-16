#!/usr/bin/env python3
"""
快速回测脚本 — 端到端下载 + 因子 + 回测 (无需数据缓存)
========================================================
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
PYTHON_BIN = "/Users/riverosa/.workbuddy/binaries/python/envs/default/bin/python3"

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


def download_and_build(n_stocks: int = 30, kline_days: int = 1000) -> tuple:
    """
    下载微盘股 K 线数据, 构建因子面板和收益率面板

    Returns:
        (factor_panel, return_panel)
    """
    print(f"1. 获取微盘股列表 (前 {n_stocks} 只)...")
    raw = _run_westock("sector pt02GN2282 --limit 100", timeout=30)
    all_codes = parse_sector_table(raw)
    # 过滤北交所 (bj 开头, 流动性差)
    codes = [c for c in all_codes if not c.startswith('bj')][:n_stocks]
    print(f"  获取到 {len(codes)} 只 (过滤北交所)")

    # 获取实时行情 (含 mcap, pb, turnover)
    print(f"2. 获取实时行情 (含 mcap/pb/turnover)...")
    quote_data = {}
    for i in range(0, len(codes), 20):
        batch = ",".join(codes[i:i+20])
        raw_q = _run_westock(f"quote {batch}", timeout=30)
        # 解析 quote 输出
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
                    # 从 quote 中提取 mcap, pb, pe, turnover
                    # 格式不固定, 尝试按列位置
                    try:
                        mcap = float(parts[7].replace(',','').replace('亿','')) if len(parts) > 7 else 10.0
                        turnover = float(parts[6].replace('%','')) if len(parts) > 6 else 2.0
                        pb = float(parts[8]) if len(parts) > 8 else 3.0
                        pe = float(parts[9]) if len(parts) > 9 else 50.0
                    except:
                        mcap, turnover, pb, pe = 10.0, 2.0, 3.0, 50.0
                    quote_data[code] = {'mcap': mcap, 'turnover': turnover, 'pb': pb, 'pe': pe}
        time.sleep(0.3)

    # 下载 K 线
    print(f"3. 下载 K 线 (每只 {kline_days} 天)...")
    all_dfs = []
    success = 0
    for i, code in enumerate(codes):
        raw_k = _run_westock(f"kline {code} --period day --limit {kline_days} --fq qfq", timeout=30)
        df = parse_kline_table(raw_k)
        if len(df) > 60:
            df['symbol'] = code
            # 附带实时行情数据
            q = quote_data.get(code, {})
            df['mcap'] = q.get('mcap', 10.0)
            df['pb'] = q.get('pb', 3.0)
            df['turnover'] = q.get('turnover', 2.0)
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

    # 计算因子
    print("4. 计算因子...")
    factor_panel, return_panel = compute_factors(combined, mcap_pb_data=None)

    print(f"  因子面板: {factor_panel.shape}")
    print(f"  收益面板: {return_panel.shape}")
    print(f"  日期范围: {factor_panel.index.get_level_values(0).min().strftime('%Y-%m-%d')} ~ "
          f"{factor_panel.index.get_level_values(0).max().strftime('%Y-%m-%d')}")

    return factor_panel, return_panel


def run_all_backtests(factor_panel, return_panel):
    """运行全部 11 策略回测"""
    print("\n5. 运行回测...")
    print("=" * 100)
    print(f"  {'策略':<30} {'年化收益':>8}  {'夏普':>6}  {'回撤':>7}  {'胜率':>6}  {'换手率':>6}  {'终值倍数':>8}")
    print("  " + "-" * 90)

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
                commission=0.00125,
                slippage=0.002,
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

    # 汇总
    if results:
        print("\n  " + "=" * 90)
        best_sharpe = max(results, key=lambda x: x[1].sharpe_ratio)
        best_return = max(results, key=lambda x: x[1].annual_return)
        best_calmar = max(results, key=lambda x: x[1].calmar_ratio)
        print(f"  最高夏普: {best_sharpe[0]} → {best_sharpe[1].annual_return}% / Sharpe {best_sharpe[1].sharpe_ratio}")
        print(f"  最高收益: {best_return[0]} → {best_return[1].annual_return}% / 终值 {best_return[1].terminal_value}x")
        print(f"  最高Calmar: {best_calmar[0]} → {best_calmar[1].annual_return}% / 回撤 {best_calmar[1].max_drawdown}%")

    return results


if __name__ == "__main__":
    n = 30
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    factor_panel, return_panel = download_and_build(n_stocks=n, kline_days=1000)
    if factor_panel is not None:
        results = run_all_backtests(factor_panel, return_panel)
