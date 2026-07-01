import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from tools.backtest_mvp.factors.combiner import combine_factors, make_composite_strategy_def


def panel_index(n_dates=36, n_symbols=30):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    return pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])


def additive_data(n_dates=40, n_symbols=35):
    idx = panel_index(n_dates, n_symbols)
    date_codes = pd.Series(idx.get_level_values("date")).factorize()[0]
    sym_codes = pd.Series(idx.get_level_values("symbol")).factorize()[0]
    s1 = pd.Series(np.sin(sym_codes * 0.7) + 0.01 * date_codes, index=idx)
    s2 = pd.Series(np.cos(sym_codes * 0.5 + date_codes * 0.2), index=idx)
    ret = s1 + s2
    return idx, ret, s1, s2


def ic_of(x, y):
    vals = []
    for _, g in pd.DataFrame({"x": x, "y": y}).dropna().groupby(level="date"):
        vals.append(g["x"].corr(g["y"], method="spearman"))
    return float(pd.Series(vals).mean())


def test_decorrelation_drops_near_duplicate_and_keeps_stronger_first():
    idx, ret, s1, s2 = additive_data()
    factors = {"good": ret, "dup": ret * 1.001 + 0.0001, "other": s2}
    res = combine_factors(factors, ret, max_corr=0.8)
    assert "good" in res.selected
    assert "dup" not in res.selected


def test_negative_ic_factor_is_sign_aligned_positive_composite_ic():
    idx, ret, *_ = additive_data()
    res = combine_factors({"neg": -ret}, ret, method="equal")
    assert res.ic["neg"] < 0
    assert ic_of(res.composite, ret) > 0.95


@pytest.mark.parametrize("method", ["equal", "ic", "ic_ir"])
def test_weight_methods_normalize(method):
    idx, ret, s1, s2 = additive_data()
    res = combine_factors({"a": s1, "b": s2}, ret, method=method, max_corr=0.99)
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert set(res.weights) == set(res.selected)


def test_ic_ir_weights_more_than_equal_for_higher_ir_factor():
    idx, ret, s1, s2 = additive_data()
    strong = ret
    weak = 0.25 * s1 + 0.75 * pd.Series(np.tile(np.arange(35), 40), index=idx)
    res = combine_factors({"strong": strong, "weak": weak}, ret, method="ic_ir", max_corr=0.99)
    assert res.weights["strong"] > 0.5


def test_composite_quality_beats_selected_constituents_for_additive_signals():
    idx, ret, s1, s2 = additive_data()
    res = combine_factors({"s1": s1, "s2": s2}, ret, method="equal", max_corr=0.99)
    best = max(ic_of(s1, ret), ic_of(s2, ret))
    assert ic_of(res.composite, ret) >= best


def test_empty_input_graceful():
    res = combine_factors({}, pd.Series(dtype=float))
    assert res.selected == []
    assert res.composite.empty
    assert res.weights == {}


def test_single_factor_equals_sign_aligned_standardized_factor():
    idx, ret, *_ = additive_data()
    res = combine_factors({"only": -ret}, ret, method="equal")
    expected = ret.groupby(level="date", group_keys=False).apply(lambda s: (s - s.mean()) / s.std())
    pd.testing.assert_series_equal(res.composite, expected, check_names=False)
    assert res.weights == {"only": 1.0}


def test_all_noise_no_crash_and_valid_weights():
    idx = panel_index()
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(size=len(idx)), index=idx)
    factors = {"n1": pd.Series(rng.normal(size=len(idx)), index=idx), "n2": pd.Series(rng.normal(size=len(idx)), index=idx)}
    res = combine_factors(factors, ret, max_corr=0.9)
    assert set(res.selected).issubset(factors)
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert not res.composite.empty


def test_nan_handling_keeps_non_nan_composite():
    idx, ret, s1, s2 = additive_data()
    ret = ret.copy(); s1 = s1.copy(); s2 = s2.copy()
    ret.iloc[::7] = np.nan
    s1.iloc[::5] = np.nan
    s2.iloc[::11] = np.nan
    res = combine_factors({"s1": s1, "s2": s2}, ret, max_corr=0.99)
    assert res.composite.notna().any()
    assert not res.composite.isna().all()


def test_corr_matrix_square_symmetric_diagonal_one():
    idx, ret, s1, s2 = additive_data()
    res = combine_factors({"s1": s1, "s2": s2}, ret, max_corr=0.99)
    corr = res.corr_matrix
    assert corr.shape == (len(res.selected), len(res.selected))
    pd.testing.assert_frame_equal(corr, corr.T)
    assert np.diag(corr).tolist() == pytest.approx([1.0] * len(res.selected))


def make_price_panel(n_dates=70, n_symbols=16):
    idx = panel_index(n_dates, n_symbols)
    dates = idx.get_level_values("date")
    symbols = idx.get_level_values("symbol")
    d_code = pd.Series(dates).factorize()[0]
    s_code = pd.Series(symbols).factorize()[0]
    signal = np.sin(s_code * 0.9) + np.cos(d_code * 0.2)
    close = 100 + d_code * 0.1 + np.cumsum(0.001 * signal)
    panel = pd.DataFrame({
        "close": close,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "volume": 1000 + s_code,
        "amount": close * (1000 + s_code),
        "vwap": close,
        "mcap": 20 + s_code,
        "pb": 1 + s_code / 100,
        "turnover": 1.0,
        "shareholders": 10000,
        "industry_code": s_code % 3,
    }, index=idx)
    panel["signal"] = signal
    return panel


def test_make_composite_strategy_def_no_lookahead_ranking_unchanged_after_future_corruption():
    panel = make_price_panel()
    current_date = panel.index.get_level_values("date").unique()[50]

    def f_signal(p):
        return p["signal"]

    def f_value(p):
        return -p["mcap"]

    strategy = make_composite_strategy_def({"signal": f_signal, "value": f_value}, lookback=30, fwd_period=5, max_corr=0.99)
    snap = panel.xs(current_date, level="date")
    snap.attrs["panel"] = panel
    snap.attrs["date"] = current_date
    base = strategy["ranking_fn"](snap)

    corrupted = panel.copy()
    future_mask = corrupted.index.get_level_values("date") > current_date
    rng = np.random.default_rng(123)
    corrupted.loc[future_mask, "signal"] = rng.normal(size=future_mask.sum())
    corrupted.loc[future_mask, "close"] = rng.normal(100, 10, size=future_mask.sum())
    snap2 = corrupted.xs(current_date, level="date")
    snap2.attrs["panel"] = corrupted
    snap2.attrs["date"] = current_date
    changed = strategy["ranking_fn"](snap2)
    pd.testing.assert_series_equal(base.sort_index(), changed.sort_index())


def test_invalid_method_raises():
    idx, ret, *_ = additive_data()
    with pytest.raises(ValueError):
        combine_factors({"x": ret}, ret, method="bad")


def test_ic_weights_proportional_to_known_ic_ratio():
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    symbols = [f"S{i}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    x = np.tile(np.arange(20), 12)
    ret = pd.Series(x, index=idx)
    high = pd.Series(x, index=idx)
    # Per-date ranks produce Spearman ICs approximately 1.0 and 0.5.
    low_pattern = np.array([0, 1, 2, 3, 4, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 16, 17, 18, 19])
    low = pd.Series(np.tile(low_pattern, 12), index=idx)
    res = combine_factors({"high": high, "low": low}, ret, method="ic", max_corr=0.99)
    total_ic = abs(res.ic["high"]) + abs(res.ic["low"])
    assert res.weights["high"] == pytest.approx(abs(res.ic["high"]) / total_ic)
    assert res.weights["low"] == pytest.approx(abs(res.ic["low"]) / total_ic)
