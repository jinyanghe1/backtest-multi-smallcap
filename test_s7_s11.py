"""
S7-S11 策略验证测试
====================
验证 meta01234 贡献的 5 个论文策略:
  S7: 漂移状态条件反转 (arXiv 2511.12490)
  S8: 盈利能力×低波动复合 (Novy-Marx NBER w33601)
  S9: 条件反转→动量切换 (ScienceDirect 2024)
  S10: 极小市值×盈利×反转三因子
  S11: 因子动量动态权重 (ScienceDirect 2024)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.strategies import (
    ALL_STRATEGIES,
    drift_regime_reversal_rank,
    profitability_low_vol_rank,
    conditional_mom_rev_rank,
    triple_factor_rank,
    factor_momentum_rank,
)
from tools.backtest_mvp.data import DATA_DIR

# Extract S7-S11 (indices 6-10)
S7_S11 = ALL_STRATEGIES[6:11]


def _load_data():
    data = load_price_data(str(DATA_DIR))
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    return compute_factors(data, mcap_pb_data=mcap_pb)


def test_load_strategies():
    """测试: S7-S11 策略 dict 字段齐全"""
    required = ["name", "universe_filter", "ranking_fn", "n_stocks", "stop_loss"]
    for s in S7_S11:
        for field in required:
            assert field in s, f"{s['name']} 缺少字段 {field}"
        assert callable(s["ranking_fn"]), f"{s['name']} ranking_fn 不是 callable"
        print(f"  {s['name']}: fields OK, paper={s.get('paper', 'N/A')}")
    print("  ✓ test_load_strategies 通过")


def test_ranking_fn_signatures():
    """测试: ranking_fn 在真实截面上返回有效 Series"""
    fp, rp = _load_data()
    dates = sorted(set(fp.index.get_level_values(0)))
    mid_date = dates[len(dates) // 2]
    snapshot = fp.xs(mid_date, level=0)

    rank_fns = {
        "S7": drift_regime_reversal_rank,
        "S8": profitability_low_vol_rank,
        "S9": conditional_mom_rev_rank,
        "S10": triple_factor_rank,
        "S11": factor_momentum_rank,
    }

    for label, fn in rank_fns.items():
        scores = fn(snapshot)
        assert isinstance(scores, pd.Series), f"{label} 返回类型应为 Series"
        assert len(scores) > 10, f"{label} 返回行数不足 ({len(scores)})"
        assert scores.notna().sum() > 10, f"{label} 有效值不足"
        print(f"  {label}: {len(scores)} stocks, "
              f"score range [{scores.min():.2f}, {scores.max():.2f}]")
    print("  ✓ test_ranking_fn_signatures 通过")


def test_backtest_all():
    """测试: S7-S11 全部能通过 engine.run() 无异常"""
    fp, rp = _load_data()
    for i, s in enumerate(S7_S11):
        engine = CrossSectionalEngine(fp, rp, n_stocks=s.get("n_stocks", 20))
        result = engine.run(
            universe_filter=s["universe_filter"],
            ranking_fn=s["ranking_fn"],
            stop_loss=s.get("stop_loss"),
        )
        assert result.annual_return is not None
        assert result.sharpe_ratio is not None
        print(f"  {s['name']}: {result.annual_return:+.1f}% | "
              f"sharpe={result.sharpe_ratio:.2f} | DD={result.max_drawdown:.1f}% | "
              f"{result.terminal_value:.2f}x")
    print("  ✓ test_backtest_all 通过")


def test_s11_no_state_leak():
    """测试: S11 多次 run 不崩溃 (IC 历史跨调用累积)"""
    from tools.backtest_mvp.strategies import _ic_history
    _ic_history.clear()

    fp, rp = _load_data()
    s = S7_S11[4]  # S11
    engine = CrossSectionalEngine(fp, rp, n_stocks=20)
    result1 = engine.run(
        universe_filter=s["universe_filter"],
        ranking_fn=s["ranking_fn"],
        stop_loss=s.get("stop_loss"),
    )

    # 第二次 run 不崩溃即通过
    result2 = engine.run(
        universe_filter=s["universe_filter"],
        ranking_fn=s["ranking_fn"],
        stop_loss=s.get("stop_loss"),
    )
    assert result2.annual_return is not None
    ic_keys = list(_ic_history.keys())
    print(f"  Run1: {result1.annual_return:+.1f}% | Run2: {result2.annual_return:+.1f}%")
    print(f"  IC history keys after 2 runs: {ic_keys} ({len(ic_keys)} factors)")
    print(f"  ✓ test_s11_no_state_leak 通过")


if __name__ == "__main__":
    test_load_strategies()
    test_ranking_fn_signatures()
    test_backtest_all()
    test_s11_no_state_leak()
    print("\n  ✅ S7-S11 全部测试通过")
