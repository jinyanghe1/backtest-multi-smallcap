"""Factor library: 30+ alpha factors for backtest_mvp.

Each factor is a panel-safe function that takes a MultiIndex (date, symbol)
DataFrame and returns a Series aligned with the same index.  All functions
use the existing operators (delay, rank, ts_std_dev, etc.) for consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .operators import (
    delay,
    rank,
    correlation,
    ts_std_dev,
    ts_skew,
    ts_kurt,
    ts_quantile,
    ts_rank,
    _group_apply,
    _has_level,
)


# ═══════════════════════════════════════════════════════════════════════════
# P0  Critical Factors (6)
# ═══════════════════════════════════════════════════════════════════════════

def short_term_reversal_vw(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F001: Short-term reversal (volume-weighted).

    Formula: -rank( ts_mean(daily_return * volume, window) / ts_mean(volume, window) )
    Logic: High volume + reversal = stronger overreaction correction.
    Expected IC: negative, |IC| ~ 0.03-0.06
    """
    close = panel["close"]
    volume = panel["volume"]

    # Daily return (panel-safe via groupby)
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    # Volume-weighted average return
    weighted = ret * volume
    vw_sum = _group_apply(weighted, "symbol", lambda s: s.rolling(window, min_periods=10).sum())
    vol_sum = _group_apply(volume, "symbol", lambda s: s.rolling(window, min_periods=10).sum())
    vw_ret = vw_sum / vol_sum.replace(0, np.nan)

    # Negative rank (want low VW-return = future high return)
    return -rank(vw_ret)


def lottery_avoidance(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F002: Lottery preference avoidance (composite).

    Formula: -(MAX_20d * 0.5 + rank(ivol_20d) * 0.3 + rank(skew_20d) * 0.2)
    Logic: Avoid stocks with high max daily return, high idiosyncratic vol,
           and high skewness — all lottery-like features loved by retail.
    Expected IC: negative, |IC| ~ 0.02-0.04
    """
    close = panel["close"]
    high = panel["high"]

    # MAX: max daily return within window (high/close - 1, using same-day high)
    max_daily = (high / close - 1).clip(lower=0)
    MAX_20d = _group_apply(max_daily, "symbol", lambda s: s.rolling(window, min_periods=10).max())

    # ivol: idiosyncratic volatility from market regression (β≈1 proxy)
    ret = close.groupby(level="symbol", group_keys=False).pct_change()
    market_ret = ret.groupby(level="date").transform("mean")
    residual = ret - market_ret
    ivol_20d = ts_std_dev(residual, window, group_level="symbol")

    # skew: rolling skewness of daily returns
    skew_20d = ts_skew(ret, window, group_level="symbol")

    # Composite: avoid high values for all three
    result = -(MAX_20d * 0.5 + rank(ivol_20d) * 0.3 + rank(skew_20d) * 0.2)
    return result


def idiosyncratic_volatility(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F006: Idiosyncratic volatility anomaly.

    Formula: -rank( std(residuals from market regression, window) )
    Logic: High ivol stocks have lower future returns (Ang et al. 2006).
    Expected IC: negative, |IC| ~ 0.03-0.05
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    # Market return (equal-weight cross-sectional mean)
    market_ret = ret.groupby(level="date").transform("mean")

    # Residual from β≈1 regression (microcap universe β≈1 on average)
    residual = ret - market_ret

    # Rolling std of residual
    ivol = ts_std_dev(residual, window, group_level="symbol")

    return -rank(ivol)


def max_daily_return(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F007: Maximum daily return (lottery-type discount).

    Formula: -ts_max(high/close - 1, window)
    Logic: High MAX attracts retail investors, future returns are low (Bali et al. 2011).
    Expected IC: negative, |IC| ~ 0.02-0.04
    """
    close = panel["close"]
    high = panel["high"]

    max_daily = (high / close - 1).clip(lower=0)
    MAX_20d = _group_apply(max_daily, "symbol", lambda s: s.rolling(window, min_periods=10).max())

    return -MAX_20d


def price_volume_divergence(panel: pd.DataFrame, window: int = 10) -> pd.Series:
    """F014: Price-volume divergence (rank correlation).

    Formula: -correlation( rank(close), rank(volume), window )
    Logic: Negative price-volume correlation = divergence = reversal signal.
    Expected IC: positive, IC ~ 0.02-0.04
    """
    close = panel["close"]
    volume = panel["volume"]

    # Rank within each date (cross-sectional)
    rank_close = rank(close)
    rank_volume = rank(volume)

    # Rolling correlation within each symbol
    corr = correlation(rank_close, rank_volume, window, group_level="symbol")

    return -corr


def turnover_anomaly(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F018: Turnover anomaly (abnormal increase).

    Formula: -rank( turnover_20d / delay(turnover_20d, window) - 1 )
    Logic: Sudden turnover increase = retail influx = future underperformance.
    Expected IC: negative, |IC| ~ 0.02-0.04

    Note: Uses 'turnover' column if available; otherwise estimates from volume/mcap.
    """
    if "turnover" in panel.columns:
        turnover = panel["turnover"]
    else:
        # Estimate turnover from volume / float_shares proxy
        volume = panel["volume"]
        mcap = panel["mcap"]
        # Assume avg price ≈ close, float_shares ≈ mcap / close
        float_proxy = mcap / panel["close"]
        turnover = volume / float_proxy.replace(0, np.nan)
        turnover = turnover.clip(0, 20)

    # Rolling mean turnover (smoothed)
    turnover_ma = _group_apply(turnover, "symbol", lambda s: s.rolling(window, min_periods=10).mean())

    # Change rate relative to window periods ago
    turnover_lag = delay(turnover_ma, window, group_level="symbol")
    turnover_change = turnover_ma / turnover_lag.replace(0, np.nan) - 1

    return -rank(turnover_change)


# ═══════════════════════════════════════════════════════════════════════════
# P1  High-Priority Factors (7)
# ═══════════════════════════════════════════════════════════════════════════

def shareholder_concentration(panel: pd.DataFrame) -> pd.Series:
    """F003: Shareholder concentration (institutional accumulation proxy).

    PRIMARY: -pct_change(shareholders, 1q)  [not available in our data]
    FALLBACK: -rank(turnover) + rank(ret_20d) * 0.5

    Logic: Fewer shareholders = concentrated chips = institutional buying.
    Expected IC: positive, IC ~ 0.02-0.03
    """
    if "shareholders" in panel.columns:
        shr = panel["shareholders"]
        return -shr.groupby(level="symbol", group_keys=False).pct_change(60)  # ~1 quarter

    # Fallback: low turnover + recent positive return = accumulation
    close = panel["close"]
    ret_20d = close.groupby(level="symbol", group_keys=False).pct_change(20)

    if "turnover" in panel.columns:
        turnover = panel["turnover"]
    else:
        volume = panel["volume"]
        mcap = panel["mcap"]
        float_proxy = mcap / close
        turnover = (volume / float_proxy.replace(0, np.nan)).clip(0, 20)

    return -rank(turnover) + rank(ret_20d) * 0.5


def earnings_acceleration_proxy(panel: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """F004: Earnings acceleration (PEAD proxy via price momentum).

    Formula: ret_5d / (abs(ret_20d) + 0.01) - 1
    Logic: Recent acceleration vs longer-term trend. Captures information diffusion early stage.
    Expected IC: positive, IC ~ 0.02-0.04
    """
    close = panel["close"]
    ret_5d = close.groupby(level="symbol", group_keys=False).pct_change(short_window)
    ret_20d = close.groupby(level="symbol", group_keys=False).pct_change(long_window)

    acceleration = ret_5d / (abs(ret_20d) + 0.01) - 1
    return acceleration


def quality_value(panel: pd.DataFrame) -> pd.Series:
    """F005: Quality value (avoid value traps).

    PRIMARY: -pb + rank(operating_cashflow / mcap) * 0.3
    FALLBACK: -pb (pure value)

    Logic: Cheap stocks with good cash flow quality = better future returns.
    Expected IC: positive, IC ~ 0.015-0.025
    """
    pb = panel["pb"]
    mcap = panel["mcap"]

    base = -pb

    if "operating_cashflow" in panel.columns:
        ocf = panel["operating_cashflow"]
        ocf_mcap = ocf / mcap.replace(0, np.nan)
        base = base + rank(ocf_mcap) * 0.3

    return base


def amihud_illiquidity(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F011: Amihud illiquidity (illiquidity premium).

    Formula: ts_mean( abs(daily_return) / amount, window )
    Logic: High illiquidity = high liquidity risk = future return compensation.
    Expected IC: positive, IC ~ 0.02-0.04

    Fallback: if amount is not available, use volume * close as proxy.
    """
    close = panel["close"]
    amount = panel.get("amount")
    if amount is None or amount.isna().all():
        # Proxy: amount ≈ volume * close (avg price × volume)
        volume = panel["volume"]
        amount = volume * close

    ret = close.groupby(level="symbol", group_keys=False).pct_change().abs()
    illiq = ret / amount.replace(0, np.nan)

    amihud = _group_apply(illiq, "symbol", lambda s: s.rolling(window, min_periods=10).mean())
    return amihud


def overnight_gap(panel: pd.DataFrame) -> pd.Series:
    """F012: Overnight gap (open vs previous close).

    Formula: rank( (open - delay(close,1)) / delay(close,1) )
    Logic: Large upward gap often gets filled (reversal). Negative IC expected.
    Expected IC: negative, |IC| ~ 0.015-0.03
    """
    close = panel["close"]
    open_ = panel["open"]

    prev_close = delay(close, 1, group_level="symbol")
    gap = (open_ - prev_close) / prev_close.replace(0, np.nan)

    return gap  # Negative IC: large gap → future reversal


def earnings_momentum(panel: pd.DataFrame) -> pd.Series:
    """F027: Earnings momentum (SUE proxy).

    PRIMARY: SUE = (EPS - expected_EPS) / stddev(EPS, 8q)
    FALLBACK: ret_60d (use 60-day return as earnings surprise proxy)

    Expected IC: positive, IC ~ 0.02-0.04
    """
    if "EPS" in panel.columns and "expected_EPS" in panel.columns:
        eps = panel["EPS"]
        expected = panel["expected_EPS"]
        sue = (eps - expected) / eps.groupby(level="symbol", group_keys=False).rolling(8).std().reset_index(level=0, drop=True)
        return sue

    # Fallback: 60-day return as earnings surprise proxy
    close = panel["close"]
    ret_60d = close.groupby(level="symbol", group_keys=False).pct_change(60)
    return ret_60d


def cashflow_price(panel: pd.DataFrame) -> pd.Series:
    """F028: Cash flow to price ratio.

    PRIMARY: operating_cashflow / mcap
    FALLBACK: -pb (value proxy)

    Expected IC: positive, IC ~ 0.02-0.03
    """
    if "operating_cashflow" in panel.columns and "mcap" in panel.columns:
        ocf = panel["operating_cashflow"]
        mcap = panel["mcap"]
        return ocf / mcap.replace(0, np.nan)

    # Fallback: -pb
    return -panel.get("pb", pd.Series(1.0, index=panel.index))


# ═══════════════════════════════════════════════════════════════════════════
# P2  Medium-Priority Factors (8)
# ═══════════════════════════════════════════════════════════════════════════

def accruals_quality(panel: pd.DataFrame) -> pd.Series:
    """F008: Accruals quality (Sloan 1996).

    PRIMARY: -(accruals / total_assets)
    FALLBACK: None (requires financial data)

    Expected IC: negative, |IC| ~ 0.015-0.025
    """
    if "accruals" in panel.columns and "total_assets" in panel.columns:
        accruals = panel["accruals"]
        ta = panel["total_assets"]
        return -(accruals / ta.replace(0, np.nan))
    return pd.Series(np.nan, index=panel.index)


def asset_growth(panel: pd.DataFrame, window: int = 4) -> pd.Series:
    """F009: Asset growth (Cooper et al. 2008).

    Formula: -(total_assets / delay(total_assets, 4q) - 1)
    Logic: High asset growth = over-investment = future underperformance.
    Expected IC: negative, |IC| ~ 0.02-0.03
    """
    if "total_assets" not in panel.columns:
        return pd.Series(np.nan, index=panel.index)

    ta = panel["total_assets"]
    ta_lag = delay(ta, 60, group_level="symbol")  # ~60 trading days ≈ 1 quarter
    growth = ta / ta_lag.replace(0, np.nan) - 1
    return -growth


def gross_profitability(panel: pd.DataFrame) -> pd.Series:
    """F010: Gross profitability (Novy-Marx 2013).

    Formula: gross_profit / total_assets
    Expected IC: positive, IC ~ 0.015-0.03
    """
    if "gross_profit" in panel.columns and "total_assets" in panel.columns:
        gp = panel["gross_profit"]
        ta = panel["total_assets"]
        return gp / ta.replace(0, np.nan)
    return pd.Series(np.nan, index=panel.index)


def vwap_deviation(panel: pd.DataFrame) -> pd.Series:
    """F013: VWAP deviation (intraday structure).

    Formula: (vwap - close) / close
    Logic: vwap > close = selling pressure (institutional distribution).
    Expected IC: negative
    """
    if "vwap" in panel.columns:
        vwap = panel["vwap"]
        close = panel["close"]
        return (vwap - close) / close.replace(0, np.nan)

    # Estimate vwap from amount / volume
    if "amount" in panel.columns and "volume" in panel.columns:
        amount = panel["amount"]
        volume = panel["volume"]
        vwap_est = amount / volume.replace(0, np.nan)
        close = panel["close"]
        return (vwap_est - close) / close.replace(0, np.nan)

    return pd.Series(np.nan, index=panel.index)


def volatility_ratio(panel: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.Series:
    """F015: Volatility ratio (short-term vs long-term volatility).

    Formula: -rank( vol_short / vol_long )
    Logic: Rising volatility = increasing risk = future underperformance.
    Expected IC: negative
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    vol_short = ts_std_dev(ret, short_window, group_level="symbol")
    vol_long = ts_std_dev(ret, long_window, group_level="symbol")

    ratio = vol_short / vol_long.replace(0, np.nan)
    return -rank(ratio)


def industry_momentum(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F016: Industry momentum (Moskowitz & Grinblatt 1999).

    Formula: industry_avg_ret_20d
    Logic: Leading industries continue to outperform.
    Expected IC: positive, IC ~ 0.02-0.04
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change(window)

    if "industry_code" in panel.columns:
        ind = panel["industry_code"]
        # Group by (date, industry) and take mean
        df = pd.DataFrame({"ret": ret, "industry": ind})
        ind_mom = df.groupby([df.index.get_level_values("date"), "industry"])["ret"].transform("mean")
        return ind_mom

    return pd.Series(np.nan, index=panel.index)


def disagreement_volatility(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F017: Disagreement (volatility / |mean return|).

    Formula: -rank( std(ret, window) / (abs(mean(ret, window)) + 0.001) )
    Logic: High disagreement = retail divergence = future underperformance.
    Expected IC: negative
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    std_ret = ts_std_dev(ret, window, group_level="symbol")
    mean_ret = _group_apply(ret, "symbol", lambda s: s.rolling(window, min_periods=10).mean())
    disagreement = std_ret / (abs(mean_ret) + 0.001)

    return -rank(disagreement)


def skewness_avoidance(panel: pd.DataFrame, window: int = 60) -> pd.Series:
    """F022: Skewness avoidance (lottery preference).

    Formula: -rank( skew(ret, window) )
    Logic: High skewness = lottery-like payoff = overpriced by retail.
    Expected IC: negative
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    skew = ts_skew(ret, window, group_level="symbol")
    return -rank(skew)


# ═══════════════════════════════════════════════════════════════════════════
# P3  Low-Priority / Advanced Factors (9)
# ═══════════════════════════════════════════════════════════════════════════

def close_location_value(panel: pd.DataFrame) -> pd.Series:
    """F019: Close location value (intraday position).

    Formula: (close - open) / ((high - low) + 0.001)
    Logic: +1 = close at high (strong); -1 = close at low (weak).
    Expected IC: positive
    """
    close = panel["close"]
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]

    return (close - open_) / ((high - low) + 0.001)


def drift_state_momentum(panel: pd.DataFrame, short_window: int = 5) -> pd.Series:
    """F020: Drift state momentum (conditional).

    Formula: if ts_min(ret, 5) > 0: momentum else: reversal
    Logic: Trend state = momentum; non-trend = reversal.
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    min_ret = _group_apply(ret, "symbol", lambda s: s.rolling(short_window, min_periods=3).min())
    mom = rank(close.groupby(level="symbol", group_keys=False).pct_change(20))

    # If min_ret > 0 (all positive), momentum; else reversal
    result = pd.Series(np.where(min_ret > 0, mom, -mom), index=panel.index)
    return result


def beta_arbitrage(panel: pd.DataFrame, window: int = 60) -> pd.Series:
    """F021: Beta arbitrage (low beta anomaly).

    Formula: -rank( beta_60d )
    Logic: Low beta stocks have higher risk-adjusted returns.
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    market_ret = ret.groupby(level="date").transform("mean")

    # Rolling beta (cov(ret, market) / var(market))
    cov = _group_apply(
        pd.DataFrame({"ret": ret, "mkt": market_ret}),
        "symbol",
        lambda df: df["ret"].rolling(window, min_periods=30).cov(df["mkt"])
    )
    var_mkt = market_ret.groupby(level="symbol", group_keys=False).rolling(window, min_periods=30).var().reset_index(level=0, drop=True)

    beta = cov / var_mkt.replace(0, np.nan)
    return -rank(beta)


def trend_strength(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F023: Trend strength (signal-to-noise ratio of momentum).

    Formula: rank( abs(mean(ret, window)) / std(ret, window) )
    Logic: High ratio = strong clean trend = momentum persistence.
    Expected IC: positive
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    mean_ret = _group_apply(ret, "symbol", lambda s: s.rolling(window, min_periods=10).mean())
    std_ret = ts_std_dev(ret, window, group_level="symbol")

    ratio = abs(mean_ret) / std_ret.replace(0, np.nan)
    return rank(ratio)


def accumulation_distribution(panel: pd.DataFrame) -> pd.Series:
    """F024: Accumulation/Distribution Rate (ADR).

    Formula: sign(delta(volume)) * (-delta(close))
    Logic: Volume increase + price decrease = accumulation (positive signal).
    """
    close = panel["close"]
    volume = panel["volume"]

    vol_change = np.sign(volume.groupby(level="symbol", group_keys=False).diff())
    price_change = -close.groupby(level="symbol", group_keys=False).diff()

    return vol_change * price_change


def momentum_quality(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    """F025: Momentum quality (high momentum + low volatility).

    Formula: rank(mom_20d) * rank(vol_20d)
    Logic: Strong momentum with low volatility = institutional quality trend.
    Expected IC: positive
    """
    close = panel["close"]
    mom = close.groupby(level="symbol", group_keys=False).pct_change(window)
    ret = close.groupby(level="symbol", group_keys=False).pct_change()
    vol = ts_std_dev(ret, window, group_level="symbol")

    return rank(mom) * rank(vol)


def tail_return_spread(panel: pd.DataFrame, window: int = 60) -> pd.Series:
    """F026: Tail return spread (P95 - P5 of daily returns).

    Formula: -rank( percentile(ret, 95) - percentile(ret, 5) )
    Logic: Wide tail = high extreme risk = retail-dominated = underperform.
    """
    close = panel["close"]
    ret = close.groupby(level="symbol", group_keys=False).pct_change()

    p95 = ts_quantile(ret, window, quantile=0.95, group_level="symbol")
    p05 = ts_quantile(ret, window, quantile=0.05, group_level="symbol")
    tail = p95 - p05

    return -rank(tail)


def rd_intensity(panel: pd.DataFrame) -> pd.Series:
    """F029: R&D intensity.

    Formula: R&D_expense / total_revenue
    Expected IC: positive
    """
    if "RD_expense" in panel.columns and "total_revenue" in panel.columns:
        rd = panel["RD_expense"]
        rev = panel["total_revenue"]
        return rd / rev.replace(0, np.nan)
    return pd.Series(np.nan, index=panel.index)


def analyst_revision(panel: pd.DataFrame) -> pd.Series:
    """F030: Analyst estimate revision.

    PRIMARY: (current - previous) / abs(previous)
    FALLBACK: ret_20d
    """
    if "EPS_estimate" in panel.columns and "previous_EPS_estimate" in panel.columns:
        curr = panel["EPS_estimate"]
        prev = panel["previous_EPS_estimate"]
        return (curr - prev) / prev.abs().replace(0, np.nan)

    close = panel["close"]
    return close.groupby(level="symbol", group_keys=False).pct_change(20)


# ═══════════════════════════════════════════════════════════════════════════
# Factor registry for automated discovery
# ═══════════════════════════════════════════════════════════════════════════

FACTOR_REGISTRY = {
    # P0 Critical
    "F001": short_term_reversal_vw,
    "F002": lottery_avoidance,
    "F006": idiosyncratic_volatility,
    "F007": max_daily_return,
    "F014": price_volume_divergence,
    "F018": turnover_anomaly,
    # P1 High
    "F003": shareholder_concentration,
    "F004": earnings_acceleration_proxy,
    "F005": quality_value,
    "F011": amihud_illiquidity,
    "F012": overnight_gap,
    "F027": earnings_momentum,
    "F028": cashflow_price,
    # P2 Medium
    "F008": accruals_quality,
    "F009": asset_growth,
    "F010": gross_profitability,
    "F013": vwap_deviation,
    "F015": volatility_ratio,
    "F016": industry_momentum,
    "F017": disagreement_volatility,
    "F022": skewness_avoidance,
    # P3 Low
    "F019": close_location_value,
    "F020": drift_state_momentum,
    "F021": beta_arbitrage,
    "F023": trend_strength,
    "F024": accumulation_distribution,
    "F025": momentum_quality,
    "F026": tail_return_spread,
    "F029": rd_intensity,
    "F030": analyst_revision,
}


def compute_all_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute all available factors and return as a DataFrame."""
    results = {}
    for fid, func in FACTOR_REGISTRY.items():
        try:
            result = func(panel)
            if result is not None and not result.isna().all():
                results[fid] = result
        except Exception:
            pass  # Skip factors that fail (e.g., missing financial data)
    return pd.DataFrame(results)
