"""
benchmark 模块单元测试
======================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from tools.backtest_mvp.benchmark import (
    load_benchmarks,
    compute_benchmark_stats,
    compute_excess_return,
    get_primary_benchmark,
)


def test_load():
    """测试: 加载基准指数"""
    bms = load_benchmarks()
    assert len(bms) >= 2, f"应至少加载2个基准, 实际 {len(bms)}"
    for name, df in bms.items():
        assert "close" in df.columns, f"{name} 缺少 close 列"
        assert len(df) > 200, f"{name} 数据行数不足 ({len(df)})"
        print(f"  {name}: {len(df)} rows, {df['date'].min()} ~ {df['date'].max()}")
    print("  ✓ test_load 通过")


def test_compute_stats():
    """测试: 计算基准统计"""
    bms = load_benchmarks()
    stats = compute_benchmark_stats(bms)
    assert len(stats) > 0
    for _, row in stats.iterrows():
        assert -50 < row["annual_return"] < 50, \
            f"{row['benchmark']} 年化收益异常: {row['annual_return']}%"
        assert row["annual_vol"] > 0, f"{row['benchmark']} 波动率应为正"
        print(f"  {row['benchmark']}: {row['annual_return']:+.1f}% | "
              f"vol={row['annual_vol']:.1f}% | sharpe={row['sharpe']}")
    print("  ✓ test_compute_stats 通过")


def test_excess_return():
    """测试: 超额收益计算"""
    assert compute_excess_return(42.5, 12.9) == 29.6
    assert compute_excess_return(-5.0, 10.0) == -15.0
    print("  ✓ test_excess_return 通过")


def test_primary_benchmark():
    """测试: 主基准选择"""
    df = pd.DataFrame({"benchmark": ["国证2000", "中证1000"]})
    assert get_primary_benchmark(df) == "国证2000"
    df2 = pd.DataFrame({"benchmark": ["中证1000"]})
    assert get_primary_benchmark(df2) == "中证1000"
    print("  ✓ test_primary_benchmark 通过")


if __name__ == "__main__":
    test_load()
    test_compute_stats()
    test_excess_return()
    test_primary_benchmark()
    print("\n  ✅ 全部测试通过")
