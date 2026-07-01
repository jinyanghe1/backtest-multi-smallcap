"""Tests for P5 decorrelated alpha factors F037-F042.

Mirrors the structure of test_factors_p4.py: alignment / short-history safety,
per-factor constructive monotonicity sanity, no-lookahead, low correlation with
the nearest existing factor, and compute_all_factors integration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from tools.backtest_mvp.factors.factor_library import (
    FACTOR_REGISTRY,
    compute_all_factors,
    coskewness,
    overnight_intraday_tug,
    turnover_cv,
    overnight_variance_share,
    time_under_water,
    delta_amihud,
    skewness_avoidance,
    downside_beta,
    overnight_gap,
    turnover_anomaly,
    amihud_illiquidity,
    idiosyncratic_volatility,
    trailing_max_drawdown,
)


NEW_FACTORS = [
    coskewness,
    overnight_intraday_tug,
    turnover_cv,
    overnight_variance_share,
    time_under_water,
    delta_amihud,
]


# ─────────────────────────────────────────────────────────────────────────────
# Panel builders
# ─────────────────────────────────────────────────────────────────────────────

def _panel_from_wides(**wides: pd.DataFrame) -> pd.DataFrame:
    """Build a MultiIndex panel from wide (date x symbol) frames per column.

    A `close` wide frame is required; other columns default to sensible values.
    """
    close_wide = wides["close"]
    idx = pd.MultiIndex.from_product(
        [close_wide.index, close_wide.columns], names=["date", "symbol"]
    )
    panel = pd.DataFrame(index=idx)

    def _stack(name, default):
        if name in wides and wides[name] is not None:
            s = wides[name].stack()
            s.index.names = ["date", "symbol"]
            return s.reindex(idx).astype(float)
        return pd.Series(default, index=idx, dtype=float)

    panel["close"] = _stack("close", np.nan)
    panel["open"] = _stack("open", np.nan) if "open" in wides else panel["close"]
    panel["high"] = _stack("high", np.nan) if "high" in wides else panel["close"] * 1.01
    panel["low"] = _stack("low", np.nan) if "low" in wides else panel["close"] * 0.99
    panel["volume"] = _stack("volume", 1000.0)
    panel["amount"] = _stack("amount", np.nan) if "amount" in wides else panel["close"] * panel["volume"]
    panel["vwap"] = panel["close"]
    panel["mcap"] = panel["close"] * 1_000_000
    panel["pb"] = 1.5
    panel["turnover"] = _stack("turnover", 0.01) if "turnover" in wides else pd.Series(0.01, index=idx)
    panel["shareholders"] = 1000.0
    panel["industry_code"] = "I1"
    return panel


def _panel_from_close(close_wide: pd.DataFrame) -> pd.DataFrame:
    return _panel_from_wides(close=close_wide)


def _random_panel(days: int = 900, symbols: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-01", periods=days)
    names = [f"S{i:02d}" for i in range(symbols)]
    market = rng.normal(0.0002, 0.012, (days, 1))
    idio = rng.normal(0.0001, 0.018, (days, symbols))
    close = 100 * np.exp(np.cumsum(market + idio, axis=0))
    panel = _panel_from_close(pd.DataFrame(close, index=dates, columns=names))
    panel["high"] = panel["close"] * (1.005 + rng.random(len(panel)) * 0.01)
    panel["low"] = panel["close"] * (0.995 - rng.random(len(panel)) * 0.01)
    panel["open"] = panel["close"] * (1 + rng.normal(0, 0.002, len(panel)))
    panel["volume"] = rng.integers(10_000, 100_000, len(panel)).astype(float)
    panel["amount"] = panel["volume"] * panel["close"]
    panel["turnover"] = rng.uniform(0.001, 0.05, len(panel))
    panel["pb"] = rng.uniform(0.5, 5.0, len(panel))
    panel["industry_code"] = [f"I{i % 3}" for i in range(len(panel))]
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Generic contract tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("factor", NEW_FACTORS)
def test_new_factors_return_aligned_series(factor):
    panel = _random_panel()
    result = factor(panel)
    assert isinstance(result, pd.Series)
    assert result.index.equals(panel.index)
    assert result.notna().any()


@pytest.mark.parametrize("factor", NEW_FACTORS)
def test_new_factors_handle_short_history_gracefully(factor):
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame({"A": np.linspace(10, 11, len(dates)), "B": 10.0}, index=dates)
    panel = _panel_from_close(close)
    result = factor(panel)
    assert result.index.equals(panel.index)
    assert len(result) == len(panel)


# ─────────────────────────────────────────────────────────────────────────────
# F037 coskewness: negative coskewness (crashes on turbulent market) ranks high
# ─────────────────────────────────────────────────────────────────────────────

def test_f037_negative_coskewness_scores_higher():
    rng = np.random.default_rng(11)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    r_m = rng.normal(0.0, 0.02, n)
    cols = {}
    for i in range(8):  # market-driving fillers so cross-sectional mean ~ r_m
        cols[f"M{i}"] = 100 * np.cumprod(1 + r_m + rng.normal(0, 0.002, n))
    # POS coskew: up on big |r_m| moves; NEG coskew: down on big |r_m| moves
    cols["POS"] = 100 * np.cumprod(1 + 25 * r_m ** 2)
    cols["NEG"] = 100 * np.cumprod(1 - 25 * r_m ** 2)
    close = pd.DataFrame(cols, index=dates)
    result = coskewness(_panel_from_close(close), window=60).xs(dates[-1], level="date")
    assert result["NEG"] > result["POS"]


# ─────────────────────────────────────────────────────────────────────────────
# F038 overnight-intraday tug: overnight-driven gains score above intraday-driven
# ─────────────────────────────────────────────────────────────────────────────

def test_f038_overnight_driven_scores_higher():
    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n)
    growth = 1.01 ** np.arange(n)
    close_w = 100 * growth  # both stocks share the same close path (+1%/day)
    # WINNER: all gains overnight -> open == close; intraday flat
    winner_close = close_w
    winner_open = close_w  # open_t = close_{t-1}*1.01 = close_t
    # LOSER: overnight down, intraday up (net same close)
    loser_close = close_w
    loser_open = np.concatenate([[100.0], close_w[:-1] * 0.99])
    close = pd.DataFrame({"WIN": winner_close, "LOSE": loser_close}, index=dates)
    open_ = pd.DataFrame({"WIN": winner_open, "LOSE": loser_open}, index=dates)
    panel = _panel_from_wides(close=close, open=open_,
                              high=close * 1.001, low=close * 0.999)
    result = overnight_intraday_tug(panel, window=21).xs(dates[-1], level="date")
    assert result["WIN"] > result["LOSE"]


# ─────────────────────────────────────────────────────────────────────────────
# F039 turnover CV: stable turnover (low CV) scores above erratic turnover
# ─────────────────────────────────────────────────────────────────────────────

def test_f039_stable_turnover_scores_higher():
    n = 90
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame({"STABLE": np.linspace(10, 12, n),
                          "ERRATIC": np.linspace(10, 12, n),
                          "MID": np.linspace(10, 12, n)}, index=dates)
    turnover = pd.DataFrame({
        "STABLE": np.full(n, 0.02),
        "ERRATIC": np.tile([0.002, 0.05], n // 2),
        "MID": np.tile([0.015, 0.025], n // 2),
    }, index=dates)
    panel = _panel_from_wides(close=close, turnover=turnover)
    result = turnover_cv(panel, window=30).xs(dates[-1], level="date")
    assert result["STABLE"] > result["ERRATIC"]


# ─────────────────────────────────────────────────────────────────────────────
# F040 overnight variance share: gap-driven risk ranks above range-driven risk
# ─────────────────────────────────────────────────────────────────────────────

def test_f040_gap_risk_scores_higher_than_range_risk():
    rng = np.random.default_rng(5)
    n = 80
    dates = pd.bdate_range("2024-01-01", periods=n)
    base = 100 * np.cumprod(1 + rng.normal(0, 0.001, n))
    close = pd.DataFrame({"GAP": base, "RANGE": base}, index=dates)
    # GAP: large overnight jumps (open != prev close), tiny intraday range
    gap_open = np.concatenate([[100.0], base[:-1] * (1 + rng.normal(0, 0.03, n - 1))])
    open_ = pd.DataFrame({"GAP": gap_open, "RANGE": base}, index=dates)  # RANGE open == close (no gap)
    high = pd.DataFrame({"GAP": base * 1.0005, "RANGE": base * 1.05}, index=dates)
    low = pd.DataFrame({"GAP": base * 0.9995, "RANGE": base * 0.95}, index=dates)
    panel = _panel_from_wides(close=close, open=open_, high=high, low=low)
    result = overnight_variance_share(panel, window=21).xs(dates[-1], level="date")
    assert result["GAP"] > result["RANGE"]


# ─────────────────────────────────────────────────────────────────────────────
# F041 time under water: a long submerged path ranks above a steadily rising one
# ─────────────────────────────────────────────────────────────────────────────

def test_f041_underwater_path_scores_higher():
    n = 90
    dates = pd.bdate_range("2024-01-01", periods=n)
    # UNDERWATER: peak early then stays below it for most of the window
    underwater = np.r_[np.linspace(100, 130, 15), np.linspace(129, 95, 75)]
    surface = np.linspace(100, 140, n)  # keeps setting new highs -> little time underwater
    close = pd.DataFrame({"UNDERWATER": underwater, "SURFACE": surface}, index=dates)
    result = time_under_water(_panel_from_close(close), window=60).xs(dates[-1], level="date")
    assert result["UNDERWATER"] > result["SURFACE"]


# ─────────────────────────────────────────────────────────────────────────────
# F042 delta-Amihud: rising illiquidity (recent > prior) ranks above falling
# ─────────────────────────────────────────────────────────────────────────────

def test_f042_rising_illiquidity_scores_higher():
    rng = np.random.default_rng(9)
    n = 120
    dates = pd.bdate_range("2024-01-01", periods=n)
    base = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    close = pd.DataFrame({"RISING": base, "FALLING": base}, index=dates)
    # amount low recently (last 21d) for RISING -> illiquidity rises; opposite for FALLING
    amt_rising = np.r_[np.full(n - 21, 1e8), np.full(21, 1e6)]
    amt_falling = np.r_[np.full(n - 21, 1e6), np.full(21, 1e8)]
    amount = pd.DataFrame({"RISING": amt_rising, "FALLING": amt_falling}, index=dates)
    panel = _panel_from_wides(close=close, amount=amount)
    result = delta_amihud(panel, window=21).xs(dates[-1], level="date")
    assert result["RISING"] > result["FALLING"]


# ─────────────────────────────────────────────────────────────────────────────
# No-lookahead: earlier factor values are unaffected by future rows
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "factor,kwargs",
    [
        (overnight_intraday_tug, {"window": 21}),
        (turnover_cv, {"window": 30}),
        (time_under_water, {"window": 60}),
        (delta_amihud, {"window": 21}),
    ],
)
def test_future_rows_do_not_change_earlier_factor_values(factor, kwargs):
    panel = _random_panel(days=160, symbols=5)
    baseline = factor(panel, **kwargs)
    cutoff = panel.index.get_level_values("date").unique()[100]
    corrupted = panel.copy()
    future = corrupted.index.get_level_values("date") > cutoff
    corrupted.loc[future, ["close", "high", "low", "open", "amount", "turnover"]] *= 100
    changed = factor(corrupted, **kwargs)
    pd.testing.assert_series_equal(
        baseline.loc[(cutoff, slice(None))],
        changed.loc[(cutoff, slice(None))],
        check_names=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decorrelation vs the nearest existing factor (repo convention: |corr| < 0.8;
# design target on real cross-sectional data is < 0.4).
# ─────────────────────────────────────────────────────────────────────────────

def test_new_factors_have_low_correlation_with_nearest_existing_factors():
    panel = _random_panel()
    pairs = {
        "F037-F022": (coskewness(panel), skewness_avoidance(panel)),
        "F037-F033": (coskewness(panel), downside_beta(panel)),
        "F038-F012": (overnight_intraday_tug(panel), overnight_gap(panel)),
        "F039-F018": (turnover_cv(panel), turnover_anomaly(panel)),
        "F039-F011": (turnover_cv(panel), amihud_illiquidity(panel)),
        "F040-F006": (overnight_variance_share(panel), idiosyncratic_volatility(panel)),
        "F041-F036": (time_under_water(panel), trailing_max_drawdown(panel)),
        "F042-F011": (delta_amihud(panel), amihud_illiquidity(panel)),
    }
    correlations = {}
    for name, (new_factor, old_factor) in pairs.items():
        aligned = pd.concat([new_factor, old_factor], axis=1).dropna()
        correlations[name] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    assert correlations
    assert all(abs(corr) < 0.8 for corr in correlations.values()), correlations


# ─────────────────────────────────────────────────────────────────────────────
# Registry / integration
# ─────────────────────────────────────────────────────────────────────────────

def test_p5_factors_registered():
    for fid in ["F037", "F038", "F039", "F040", "F041", "F042"]:
        assert fid in FACTOR_REGISTRY
    assert len(FACTOR_REGISTRY) == 42


def test_compute_all_factors_includes_p5():
    panel = _random_panel(days=200, symbols=6)
    df = compute_all_factors(panel)
    for fid in ["F037", "F038", "F039", "F040", "F041", "F042"]:
        assert fid in df.columns
