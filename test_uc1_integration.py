"""Quick integration test for UC1-4: verify engine.run(neutralize=True) works.

This tests the end-to-end flow: data load -> engine -> neutralize=True run.
It does NOT test the full IC improvement (that's UC1-5's job).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.strategies import strategy_micro_rotation, strategy_shell


def test_engine_with_neutralize():
    """Test that engine.run(neutralize=True) runs without error."""
    print("Loading data...")
    data = load_price_data(str(DATA_DIR))
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    fp, rp = compute_factors(data, mcap_pb_data=mcap_pb)

    print(f"Data: {len(fp)} rows, {len(fp.index.get_level_values(1).unique())} stocks")

    engine = CrossSectionalEngine(fp, rp, n_stocks=10)

    # Test S3 (micro rotation) with neutralize=True
    print("\n--- Test S3 with neutralize=True ---")
    s3 = strategy_micro_rotation
    result = engine.run(
        universe_filter=s3['universe_filter'],
        ranking_factor=s3['ranking_factor'],
        ascending=s3['ascending'],
        stop_loss=s3.get('stop_loss'),
        neutralize=True,
        neutralize_strength=0.5,
    )
    print(f"  Sharpe: {result.sharpe_ratio:.2f}, DD: {result.max_drawdown:.1f}%")
    assert result.sharpe_ratio is not None
    assert result.terminal_value > 0

    # Test S5 with neutralize=True
    print("\n--- Test S5 with neutralize=True ---")
    s5 = strategy_shell
    result2 = engine.run(
        universe_filter=s5['universe_filter'],
        ranking_factor=s5['ranking_factor'],
        ascending=s5['ascending'],
        stop_loss=s5.get('stop_loss'),
        neutralize=True,
        neutralize_strength=0.5,
    )
    print(f"  Sharpe: {result2.sharpe_ratio:.2f}, DD: {result2.max_drawdown:.1f}%")
    assert result2.sharpe_ratio is not None
    assert result2.terminal_value > 0

    print("\n✅ All integration tests passed!")


if __name__ == '__main__':
    test_engine_with_neutralize()
