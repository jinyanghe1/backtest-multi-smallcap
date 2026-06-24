"""CLI entry point for the industry package.

Usage:
    python -m tools.backtest_mvp.industry
    python -m tools.backtest_mvp.industry --limit 50
"""

from tools.backtest_mvp.industry.build_cache import main

if __name__ == "__main__":
    main()
