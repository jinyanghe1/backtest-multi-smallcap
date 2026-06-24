#!/usr/bin/env python3
"""
因子审计 + 基准对比 + IC 面板。

Import-safe: importing ``compute_ic_weights`` does not load local caches or run
the full report. Execute this file directly to print the analysis report.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = Path(__file__).resolve().parent
BENCH_DIR = PROJECT / "benchmarks"
DEFAULT_IC_FACTORS = ["mcap", "pb", "pe", "mom20d", "mom60d", "turnover", "vol20d", "ivol", "max_ret"]


def load_all():
    try:
        from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
        from tools.backtest_mvp.data import DATA_DIR
    except ModuleNotFoundError:
        from factors import load_price_data, compute_factors, load_daily_mcap_pb
        from data import DATA_DIR

    data = load_price_data(str(DATA_DIR))
    mpb = load_daily_mcap_pb(str(DATA_DIR))
    return compute_factors(data, mcap_pb_data=mpb)


def load_benchmarks():
    """加载基准指数。"""
    benchmarks = {}
    for path in BENCH_DIR.glob("*.parquet"):
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["ret"] = df["close"].pct_change()
        benchmarks[path.stem] = df["ret"]
    return benchmarks


def compute_ic_weights(
    factor_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    lookback: int = 24,
    factor_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    基于历史 rank IC 计算因子权重。

    Returns:
        {因子名: 归一化权重}，权重总和=1.0，IC 为负时取绝对值。
    """
    dates = sorted(set(factor_panel.index.get_level_values(0)))
    if factor_list is None:
        factor_list = [
            c for c in factor_panel.columns
            if c not in ("name", "is_limit_up", "is_limit_down")
        ]

    month_groups = pd.DatetimeIndex(dates).to_period("M")
    dates_arr = np.array(dates)
    ic_records = []

    for month in sorted(set(month_groups)):
        mask = month_groups == month
        m_dates = dates_arr[mask]
        if len(m_dates) < 15:
            continue

        next_month = month + 1
        next_dates = dates_arr[month_groups == next_month]
        if len(next_dates) < 10:
            continue

        try:
            f_snap = factor_panel.xs(m_dates[-1], level=0, drop_level=True)
            r = return_panel.xs(next_dates[-1], level=0, drop_level=True)
        except (KeyError, AttributeError):
            continue

        fwd_return = r["daily_return"]
        for fac in factor_list:
            if fac not in f_snap.columns:
                continue
            f_vals = f_snap[fac].dropna()
            common = f_vals.index.intersection(fwd_return.index)
            if len(common) < 30:
                continue
            ic = f_vals.loc[common].rank().corr(fwd_return.loc[common].rank(), method="pearson")
            ic_records.append({"month": str(month), "factor": fac, "IC": ic})

    if not ic_records:
        return {}

    ic_df = pd.DataFrame(ic_records)
    all_months = sorted(set(ic_df["month"]))
    recent = all_months[-lookback:] if len(all_months) >= lookback else all_months
    ic_df = ic_df[ic_df["month"].isin(recent)]

    weights = {}
    for fac in factor_list:
        sub = ic_df[ic_df["factor"] == fac]
        if len(sub) < 5:
            continue
        weights[fac] = sub["IC"].abs().mean()

    total = sum(weights.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in weights.items()}


def _compute_monthly_rank_ic(factor_panel: pd.DataFrame, return_panel: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
    dates = sorted(set(factor_panel.index.get_level_values(0)))
    dates_arr = np.array(dates)
    month_groups = pd.DatetimeIndex(dates).to_period("M")
    records = []

    for month in sorted(set(month_groups)):
        m_dates = dates_arr[month_groups == month]
        next_dates = dates_arr[month_groups == month + 1]
        if len(m_dates) < 2 or len(next_dates) == 0:
            continue

        try:
            f_snap = factor_panel.xs(m_dates[-1], level=0)
            r_next = return_panel.loc[pd.IndexSlice[next_dates, :], "daily_return"]
        except (KeyError, AttributeError):
            continue

        fwd_return = r_next.groupby(level=1).apply(lambda x: (1 + x).prod() - 1)
        for fac in factors:
            if fac not in f_snap.columns:
                continue
            f_vals = f_snap[fac].dropna()
            common = f_vals.index.intersection(fwd_return.index)
            if len(common) < 30:
                continue
            ic = f_vals.loc[common].rank().corr(fwd_return.loc[common].rank(), method="pearson")
            records.append({"month": str(month), "factor": fac, "IC": ic})

    return pd.DataFrame(records)


def print_factor_redundancy(factor_panel: pd.DataFrame, factor_cols: List[str], dates: List[pd.Timestamp]) -> None:
    print("\n" + "=" * 80)
    print("  一、因子冗余审计")
    print("=" * 80)

    latest_date = dates[-1]
    snap = factor_panel.xs(latest_date, level=0)[factor_cols].dropna(how="all")
    print(f"\n  截面日期: {latest_date.date()}")
    print(f"  有效股票: {len(snap)}")

    corr = snap.corr(method="pearson")
    print("\n  ── 冗余因子对 (|corr| > 0.7) ──")
    redundant = []
    for i in range(len(factor_cols)):
        for j in range(i + 1, len(factor_cols)):
            value = corr.iloc[i, j]
            if abs(value) > 0.7:
                redundant.append((factor_cols[i], factor_cols[j], value))
                tag = "高度冗余" if abs(value) > 0.85 else "中度重叠"
                print(f"    {factor_cols[i]} ↔ {factor_cols[j]}: r={value:+.3f}  {tag}")
    if not redundant:
        print("    (无高度冗余因子对)")


def print_benchmark_comparison(factor_panel: pd.DataFrame, return_panel: pd.DataFrame, benchmarks: Dict[str, pd.Series]) -> None:
    print("\n\n" + "=" * 80)
    print("  二、基准对比")
    print("=" * 80)

    daily_market = return_panel.groupby(level=0)["daily_return"].mean().sort_index()
    bm_series = {"等权微盘": daily_market}
    for name, bm_ret in benchmarks.items():
        aligned = bm_ret.reindex(daily_market.index).dropna()
        if len(aligned) > 100:
            bm_series[name] = aligned

    print(f"\n  {'基准':<18} {'累计收益':>9}  {'年化收益':>9}  {'年化波动':>9}  {'夏普':>7}  {'最大回撤':>9}")
    print(f"  {'─' * 68}")
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

    try:
        from strategies import ALL_STRATEGIES
        from engine import CrossSectionalEngine
    except ModuleNotFoundError:
        from tools.backtest_mvp.strategies import ALL_STRATEGIES
        from tools.backtest_mvp.engine import CrossSectionalEngine

    market_ann = (1 + daily_market.dropna()).prod() ** (252 / len(daily_market.dropna())) - 1
    print("\n  ── 策略 vs 等权微盘 超额收益 ──")
    print(f"  {'策略':<24} {'年化':>7}  {'超额α':>7}")
    print(f"  {'─' * 42}")
    for strategy in [ALL_STRATEGIES[2], ALL_STRATEGIES[4]]:
        engine = CrossSectionalEngine(factor_panel, return_panel, n_stocks=strategy.get("n_stocks", 30))
        result = engine.run(
            universe_filter=strategy["universe_filter"],
            ranking_factor=strategy["ranking_factor"],
            ascending=strategy["ascending"],
            stop_loss=strategy.get("stop_loss"),
        )
        alpha = result.annual_return / 100 - market_ann
        print(f"  {strategy['name']:<24} {result.annual_return:>5.1f}%  {alpha * 100:>+5.1f}%")


def print_ic_panel(factor_panel: pd.DataFrame, return_panel: pd.DataFrame, factors: List[str]) -> None:
    print("\n\n" + "=" * 80)
    print("  三、因子 Rank IC 面板")
    print("=" * 80)

    ic_df = _compute_monthly_rank_ic(factor_panel, return_panel, factors)
    if ic_df.empty:
        print("  IC 计算失败 (数据不足)")
        return

    print(f"\n  ── 月频 Rank IC 统计 ({ic_df['month'].nunique()} 个月) ──")
    print(f"  {'因子':<12} {'IC均值':>8}  {'IC中位':>7}  {'IC_IR':>7}  {'IC>0%':>6}  {'t值':>6}  {'显著性'}")
    print(f"  {'─' * 65}")

    for fac in factors:
        sub = ic_df[ic_df["factor"] == fac]
        if len(sub) < 5:
            continue
        mean_ic = sub["IC"].mean()
        med_ic = sub["IC"].median()
        ic_std = sub["IC"].std()
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0
        pos_rate = (sub["IC"] > 0).mean()
        t_stat = mean_ic / (ic_std / np.sqrt(len(sub))) if ic_std > 0 else 0
        sig = "★★★" if abs(t_stat) > 2.58 else ("★★" if abs(t_stat) > 1.96 else ("★" if abs(t_stat) > 1.64 else "—"))
        print(f"  {fac:<12} {mean_ic:>+7.4f}  {med_ic:>+7.4f}  {ic_ir:>+6.2f}  {pos_rate:>5.1%}  {t_stat:>+6.2f}  {sig}")


def main() -> None:
    print("加载因子面板...", end=" ", flush=True)
    factor_panel, return_panel = load_all()
    factor_cols = [c for c in factor_panel.columns if c not in ("name", "is_limit_up", "is_limit_down")]
    dates = sorted(factor_panel.index.get_level_values(0).unique())
    stocks = sorted(factor_panel.index.get_level_values(1).unique())
    print(f"{len(stocks)} 只, {len(dates)} 交易日")

    print("加载基准指数...", end=" ", flush=True)
    benchmarks = load_benchmarks()
    print(f"{len(benchmarks)} 个基准")

    available_ic_factors = [c for c in DEFAULT_IC_FACTORS if c in factor_panel.columns]
    print_factor_redundancy(factor_panel, factor_cols, dates)
    print_benchmark_comparison(factor_panel, return_panel, benchmarks)
    print_ic_panel(factor_panel, return_panel, available_ic_factors)

    print("\n\n" + "=" * 80)
    print("  总结")
    print("=" * 80)
    print(f"""
  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  数据窗口: {dates[0].date()} ~ {dates[-1].date()}
  有效股票: {len(stocks)} 只
  因子数:   {len(factor_cols)} 个
""")


if __name__ == "__main__":
    main()
