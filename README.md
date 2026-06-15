# backtest-multi-smallcap

A-share micro-cap multi-factor cross-sectional backtest engine.

> 面向 A 股微盘股（市值 < 30 亿）的多因子截面回测系统。月度调仓、N 只等权、因子排序选股。

## Why this exists

Existing backtesting frameworks (backtrader, zipline, vnpy) are all event-driven — they feed one bar at a time to one strategy. Cross-sectional strategies (rank hundreds of stocks by mcap/PB/momentum, pick top N, rebalance monthly) require a fundamentally different paradigm: **a factor panel + a walk-forward loop over rebalance dates**.

This engine does exactly that.

## Design

```
Data Layer           Factor Layer          Engine Layer          Strategy Layer
─────────────────────────────────────────────────────────────────────────────
westock-data CLI     momentum (20d/60d)    CrossSectionalEngine  ST turnaround
  ↓                  volatility (20d)        ├─ get snapshots    Momentum + turnover
Parquet cache        mcap (static*)          ├─ universe filter  Micro-cap rotation
  ↓                  pb (static*)            ├─ factor ranking   IPO resonance
akshare (planned)    turnover                ├─ position sizing  Shell value
                                            ├─ equity tracking  PB value rotation
                                            └─ performance stats
```

Each strategy is a pure function: `(factor_snapshot, dates, step) → List[str]` — then the engine handles the rest.

## Quick Start

```bash
# Install
pip install pandas numpy openpyxl

# Download micro-cap data (30 stocks × 6 years)
python run.py download --n 30

# Check cache
python run.py status

# Run all 6 strategies
python run.py backtest

# Run single strategy
python run.py backtest --strategy 3
```

## Output

```
===============================================================================================
  六大策略回测对比 (基于真实 A 股数据)
===============================================================================================
  策略                           年化收益      夏普       回撤      胜率     换手率      终值倍数
  -------------------------------------------------------------------------------------
  策略2: 小市值动量+高换手            -6.80%   -0.21    -73.82%    43.5%     2.9%      0.76x
  策略3: 极小小市值轮动              -11.57%   -0.51    -66.77%    50.7%     2.6%      0.47x
  ...
```

## Six Strategies

| # | Name | Universe | Signal | Positions |
|---|------|----------|--------|-----------|
| 1 | ST Turnaround | ST stocks, mcap < 20B | 摘帽 + turnaround momentum | 5 |
| 2 | Momentum + Turnover | mcap < 30B, non-ST | 20d return + turnover > 3% | 8 |
| 3 | Micro-cap Rotation | mcap < 30B, non-ST | Smallest 30 by mcap | 30 |
| 4 | IPO Resonance | mcap < 30B, high turnover | 20d momentum | 6 |
| 5 | Shell Value | mcap < 12B, PB < 2 | Cheapest by PB | 10 |
| 6 | PB Value Rotation | mcap < 30B, PB < 1.5 | Cheapest by PB, 50% take-profit | 20 |

## Limitations (current)

- **mcap/PB are static** — fetched once from current quotes, not historical. Factor ranking is approximate at best. See [Roadmap](#roadmap).
- **30 stock sample** — the full micro-cap universe is ~700 stocks. Sample size limits statistical significance.
- **No T+1 / price limit simulation** — assumes all orders fill at close price.
- **No benchmark tracking** — no comparison against CSI 1000 or micro-cap index.

## Roadmap

| Phase | Task | Effort |
|-------|------|--------|
| 0 | Data availability test (10 stocks × 100 days) | ✅ Done |
| 1 | Historical fundamental data (akshare quarterly mcap/PB) | 2 days |
| 2 | Expand universe to 500+ stocks | 0.5 day |
| 3 | T+1 settlement + price limit simulation | 1 day |
| 4 | Benchmark integration (CSI 1000 / 万得微盘股指数) | 0.5 day |
| 5 | Factor IC analysis + parameter grid search | 1 day |
| 6 | Hong Kong small-cap extension | 2 days |

## File Structure

```
backtest-multi-smallcap/
├── engine.py         # CrossSectionalEngine — the core (300 lines)
├── strategies.py     # 6 strategy definitions (180 lines)
├── factors.py        # Factor computation from OHLCV (120 lines)
├── data.py           # Data download via westock-data CLI → Parquet (160 lines)
├── run.py            # CLI runner (120 lines)
├── data_cache/       # Parquet files (gitignored)
└── .gitignore
```

## Dependencies

- Python ≥ 3.10
- pandas, numpy, openpyxl
- westock-data CLI (for data fetching)
- ak share (planned, for historical fundamentals)

## Disclaimer

Academic research only. Not investment advice. A-shares are risky; micro-caps are extremely risky; ST stocks carry delisting risk. Past performance does not guarantee future results.
