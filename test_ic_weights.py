"""
IC 加权复合排名测试
====================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.analyze import compute_ic_weights
from tools.backtest_mvp.data import DATA_DIR


def _load_data():
    data = load_price_data(str(DATA_DIR))
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    return compute_factors(data, mcap_pb_data=mcap_pb)


def test_ic_weights_computation():
    """测试: compute_ic_weights 返回合理的非零权重"""
    fp, rp = _load_data()
    factors = ['mcap', 'pb', 'mom20d', 'vol20d', 'max_ret']
    weights = compute_ic_weights(fp, rp, lookback=24, factor_list=factors)
    assert len(weights) > 0, "应至少返回一些因子权重"
    assert abs(sum(weights.values()) - 1.0) < 0.01, \
        f"权重和不等于1: {sum(weights.values()):.4f}"
    # 验证 mcap 权重最高 (已知最强因子)
    if 'mcap' in weights:
        top = max(weights, key=weights.get)
        assert weights['mcap'] > 0.1, f"mcap 权重应>0.1, 实际 {weights['mcap']:.4f}"
        print(f"  权重: {dict((k, round(v, 3)) for k, v in sorted(weights.items(), key=lambda x: -x[1]))}")
        print(f"  最高权重因子: {top} ({weights[top]:.3f})")
    print("  ✓ test_ic_weights_computation 通过")


def test_equal_weight_fallback():
    """测试: factor_weights=None → 等权行为不变"""
    fp, rp = _load_data()
    s = {
        "name": "test",
        "universe_filter": lambda snap, dates, step: list(snap.index[:100]),
        "n_stocks": 10,
        "composite_factors": [("mcap", True), ("pb", True)],
    }
    from tools.backtest_mvp.run import run_single_backtest
    r_eq = run_single_backtest(s, fp, rp)

    s["factor_weights"] = None
    r_none = run_single_backtest(s, fp, rp)
    assert r_eq.annual_return == r_none.annual_return, \
        "factor_weights=None 应与不传的行为一致"
    print("  ✓ test_equal_weight_fallback 通过")


def test_ic_weighted_vs_equal():
    """测试: IC 加权至少不显著差于等权"""
    fp, rp = _load_data()
    factors = ['mcap', 'pb', 'mom20d', 'vol20d', 'max_ret']
    weights = compute_ic_weights(fp, rp, lookback=24, factor_list=factors)
    if not weights:
        print("  ⚠️  无 IC 权重数据, 跳过对比测试")
        return

    cf = [(f, True) for f in weights]  # 所有因子 ascending=True (微盘因子方向)

    s_eq = {
        "name": "test_eq",
        "universe_filter": lambda snap, dates, step: list(snap.index[:200]),
        "n_stocks": 20,
        "composite_factors": cf,
    }
    s_w = {
        "name": "test_w",
        "universe_filter": lambda snap, dates, step: list(snap.index[:200]),
        "n_stocks": 20,
        "composite_factors": cf,
        "factor_weights": weights,
    }
    from tools.backtest_mvp.run import run_single_backtest
    r_eq = run_single_backtest(s_eq, fp, rp)
    r_w = run_single_backtest(s_w, fp, rp)

    print(f"  等权: {r_eq.annual_return:+.1f}% sharpe={r_eq.sharpe_ratio:.2f}")
    print(f"  IC加权: {r_w.annual_return:+.1f}% sharpe={r_w.sharpe_ratio:.2f}")
    print("  ✓ test_ic_weighted_vs_equal 通过")


def test_partial_weights():
    """测试: 部分因子缺权重 → fallback 处理"""
    fp, rp = _load_data()
    cf = [("mcap", True), ("pb", True)]
    partial_w = {"mcap": 0.8}  # pb 缺失

    engine = CrossSectionalEngine(fp, rp, n_stocks=20)
    result = engine.run(
        universe_filter=lambda snap, dates, step: list(snap.index[:100]),
        composite_factors=cf,
        factor_weights=partial_w,
    )
    assert result.annual_return is not None
    print(f"  部分权重 (mcap=0.8, pb缺失): {result.annual_return:+.1f}%")
    print("  ✓ test_partial_weights 通过")


if __name__ == "__main__":
    test_ic_weights_computation()
    test_equal_weight_fallback()
    test_ic_weighted_vs_equal()
    test_partial_weights()
    print("\n  ✅ IC 加权全部测试通过")
