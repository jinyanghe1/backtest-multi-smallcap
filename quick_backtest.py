"""
Quick backtest with expanded universe - single strategy only
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.backtest_mvp.factors.legacy import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.engine import CrossSectionalEngine

print("="*60)
print("Quick Backtest - Expanded Universe (5,400+ stocks)")
print("="*60)

print("\n[1/4] Loading price data...")
data = load_price_data(str(DATA_DIR))
print(f"  ✓ {data['symbol'].nunique()} stocks, {len(data):,} rows")
print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")

print("\n[2/4] Loading mcap/pb data...")
mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
if not mcap_pb.empty:
    print(f"  ✓ {mcap_pb['symbol'].nunique()} stocks with mcap/pb")
else:
    print("  ⚠️ No mcap/pb data, using static fallback")

print("\n[3/4] Computing factors...")
factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)
print(f"  ✓ Factor panel: {len(factor_panel):,} rows x {len(factor_panel.columns)} cols")
print(f"  Stocks: {factor_panel.index.get_level_values(1).nunique()}")
print(f"  Date range: {factor_panel.index.get_level_values(0).min().date()} to {factor_panel.index.get_level_values(0).max().date()}")
print(f"  Factors: {list(factor_panel.columns)}")

print("\n[4/4] Running backtest (MCAP ascending - microcap strategy)...")
engine = CrossSectionalEngine(
    factor_panel=factor_panel,
    return_panel=return_panel,
    initial_capital=1.0,
    n_stocks=30,
    rebalance_freq='M',
    commission=0.00125,
    slippage=0.002,
    price_limit_stocks=True,
)

result = engine.run(
    universe_filter=lambda snapshot, dates, i: list(snapshot[snapshot['mcap'] < 50].index),
    ranking_factor='mcap',
    ascending=True,
)

print(f"\n{'='*60}")
print("RESULTS:")
print(f"  Annual Return:     {result.annual_return:>7.2f}%")
print(f"  Sharpe Ratio:      {result.sharpe_ratio:>7.2f}")
print(f"  Max Drawdown:      {result.max_drawdown:>7.2f}%")
print(f"  Win Rate:          {result.win_rate:>7.1f}%")
print(f"  Avg Turnover:      {result.avg_turnover:>7.1f}%")
print(f"  Terminal Value:    {result.terminal_value:>7.2f}x")
print(f"  IC Mean:           {result.ic_mean:>7.4f}")
print(f"  IC IR:             {result.ic_ir:>7.4f}")
print(f"{'='*60}")
