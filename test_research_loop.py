import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from tools.backtest_mvp.research_loop import AlphaCandidate, ResearchLoop, validate_metrics
from tools.backtest_mvp.research_loop.validators import (
    validate_signal_coverage,
    factor_decay_halflife,
    walk_forward_validation_split,
)


def _panels():
    dates = pd.bdate_range("2024-01-01", periods=90)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    signal = [3.0, 2.0, 1.0] * len(dates)
    factor_panel = pd.DataFrame({"alpha_signal": signal}, index=idx)
    returns = []
    for _date in dates:
        returns.extend([0.002, 0.001, -0.001])
    return_panel = pd.DataFrame({"daily_return": returns}, index=idx)
    return factor_panel, return_panel


def test_research_loop_simulate_updates_candidate_metrics():
    factor_panel, return_panel = _panels()
    alpha = AlphaCandidate(expr="alpha_signal")
    loop = ResearchLoop(n_stocks=1, rebalance_freq="M")
    result = loop.simulate(alpha, factor_panel, return_panel, ascending=False)

    assert result.terminal_value > 0
    assert "sharpe" in alpha.metrics
    assert alpha.turnover >= 0


def test_validate_metrics_uses_percent_units():
    factor_panel, return_panel = _panels()
    loop = ResearchLoop(n_stocks=1, rebalance_freq="M")
    result = loop.simulate(AlphaCandidate(expr="alpha_signal"), factor_panel, return_panel)
    validation = validate_metrics(result, thresholds={
        "sharpe": -10,
        "fitness": -10,
        "turnover": (0, 100),
        "drawdown": 100,
        "self_correlation": 0.7,
    })

    assert validation.passed


def test_signal_coverage_validator_ok():
    factor_panel, _ = _panels()
    result = validate_signal_coverage(factor_panel, "alpha_signal")
    assert result["status"] == "ok"
    assert result["coverage"] == 1.0


def test_signal_coverage_validator_warning():
    """Coverage between 10% and 30% should be 'warning'."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    # 20% non-null: 1 in 5
    vals = [3.0, np.nan, np.nan, np.nan, np.nan] * 60  # 300 total, 60 non-null = 20%
    panel = pd.DataFrame({"sparse_signal": vals}, index=idx)
    result = validate_signal_coverage(panel, "sparse_signal")
    assert result["status"] == "warning"
    assert 0.1 <= result["coverage"] < 0.3


def test_signal_coverage_validator_error():
    """Coverage below 10% should be 'error'."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    vals = [np.nan] * 300
    vals[0] = 1.0  # only 1 non-null out of 300
    panel = pd.DataFrame({"rare_signal": vals}, index=idx)
    result = validate_signal_coverage(panel, "rare_signal")
    assert result["status"] == "error"
    assert result["coverage"] < 0.1


def test_signal_coverage_missing_column():
    panel, _ = _panels()
    result = validate_signal_coverage(panel, "nonexistent")
    assert result["status"] == "error"


# ── T03: factor_decay_halflife ──

def test_factor_decay_halflife_basic():
    """IC decays linearly: 0.10 at lag 1, 0.08 at lag 5, 0.04 at lag 10.
    Half of 0.10 = 0.05. Should find lag between 5 and 10."""
    decay = {1: 0.10, 5: 0.08, 10: 0.04}
    hl = factor_decay_halflife(decay)
    assert hl is not None
    assert 5 < hl < 10


def test_factor_decay_halflife_exact_half():
    """IC hits exactly half at lag 10."""
    decay = {1: 0.10, 5: 0.08, 10: 0.05}
    hl = factor_decay_halflife(decay)
    assert hl is not None
    assert abs(hl - 10.0) < 0.01


def test_factor_decay_halflife_never_decays():
    """IC stays above half → returns largest lag."""
    decay = {1: 0.10, 5: 0.09, 10: 0.08}
    hl = factor_decay_halflife(decay)
    assert hl == 10.0


def test_factor_decay_halflife_negative_ic():
    """Negative initial IC → returns None."""
    decay = {1: -0.05, 5: -0.03}
    hl = factor_decay_halflife(decay)
    assert hl is None


def test_factor_decay_halflife_empty():
    """Empty dict → None."""
    assert factor_decay_halflife({}) is None


def test_factor_decay_halflife_single_lag():
    """Single lag → returns that lag (can't find decay point)."""
    decay = {1: 0.10}
    hl = factor_decay_halflife(decay)
    assert hl == 1.0


# ── T10: walk_forward_validation_split ──

def test_walk_forward_split_basic():
    """60/20/20 split on 100 dates."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    symbols = ["a", "b", "c"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    panel = pd.DataFrame({"signal": range(len(idx))}, index=idx)
    train, val, test = walk_forward_validation_split(panel, 0.6, 0.2)
    # 60 dates for train, 20 for val, 20 for test
    assert len(train.index.get_level_values("date").unique()) == 60
    assert len(val.index.get_level_values("date").unique()) == 20
    assert len(test.index.get_level_values("date").unique()) == 20


def test_walk_forward_split_no_overlap():
    """Train, val, test dates should not overlap."""
    dates = pd.bdate_range("2024-01-01", periods=50)
    symbols = ["a", "b"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    panel = pd.DataFrame({"signal": range(len(idx))}, index=idx)
    train, val, test = walk_forward_validation_split(panel, 0.6, 0.2)
    train_dates = set(train.index.get_level_values("date"))
    val_dates = set(val.index.get_level_values("date"))
    test_dates = set(test.index.get_level_values("date"))
    assert not train_dates & val_dates
    assert not train_dates & test_dates
    assert not val_dates & test_dates


def test_walk_forward_split_temporal_order():
    """All train dates < all val dates < all test dates."""
    dates = pd.bdate_range("2024-01-01", periods=30)
    symbols = ["a"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    panel = pd.DataFrame({"signal": range(30)}, index=idx)
    train, val, test = walk_forward_validation_split(panel, 0.5, 0.3)
    assert train.index.get_level_values("date").max() < val.index.get_level_values("date").min()
    assert val.index.get_level_values("date").max() < test.index.get_level_values("date").min()


def test_walk_forward_split_invalid_ratios():
    """Invalid ratios should raise ValueError."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    symbols = ["a"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    panel = pd.DataFrame({"signal": range(10)}, index=idx)
    try:
        walk_forward_validation_split(panel, 0.8, 0.3)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_walk_forward_split_no_date_level():
    """Panel without date level should raise ValueError."""
    panel = pd.DataFrame({"signal": [1, 2, 3]}, index=pd.Index(["x", "y", "z"], name="symbol"))
    try:
        walk_forward_validation_split(panel)
        assert False, "Should have raised"
    except ValueError:
        pass

