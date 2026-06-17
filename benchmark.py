"""
基准指数对比模块
================
加载本地缓存的基准指数日线, 计算年化收益/波动率/夏普, 供回测结果对比。

完全独立于 engine / strategies / factors, 不引入新依赖。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

BENCH_DIR = Path(__file__).parent / "benchmarks"


def load_benchmarks(bench_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    加载所有基准指数日线

    Returns:
        {指数名: DataFrame[date, close]}
    """
    if bench_dir is None:
        bench_dir = BENCH_DIR

    benchmarks = {}
    for f in sorted(bench_dir.glob("*.parquet")):
        name = f.stem.replace("中证1000", "中证1000").replace("中证500", "中证500")
        df = pd.read_parquet(f)
        if "close" not in df.columns:
            continue
        benchmarks[name] = df
    return benchmarks


def compute_benchmark_stats(
    benchmarks: Dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    计算各基准的年化指标

    Returns:
        DataFrame: index=基准名, columns=[annual_return, annual_vol, sharpe]
    """
    stats = []
    for name, df in benchmarks.items():
        bm = df.copy()
        bm["date"] = pd.to_datetime(bm["date"])
        bm = bm.sort_values("date")

        if start_date:
            bm = bm[bm["date"] >= start_date]
        if end_date:
            bm = bm[bm["date"] <= end_date]

        if len(bm) < 60:
            continue

        bm["daily_return"] = bm["close"].pct_change()
        rets = bm["daily_return"].dropna()

        n_days = len(rets)
        cum = (1 + rets).prod()
        ann_ret = cum ** (252 / n_days) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        stats.append({
            "benchmark": name,
            "annual_return": round(ann_ret * 100, 2),
            "annual_vol": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "n_days": n_days,
            "start": str(bm["date"].min().date()),
            "end": str(bm["date"].max().date()),
        })

    return pd.DataFrame(stats)


def compute_excess_return(
    strategy_return_pct: float,
    benchmark_return_pct: float,
) -> float:
    """超额收益 (pp)"""
    return strategy_return_pct - benchmark_return_pct


def get_primary_benchmark(stats_df: pd.DataFrame) -> str:
    """返回主比较基准 (优先国证2000→中证1000)"""
    for name in ["国证2000", "中证1000"]:
        if name in stats_df["benchmark"].values:
            return name
    if len(stats_df) > 0:
        return stats_df["benchmark"].iloc[0]
    return "N/A"
