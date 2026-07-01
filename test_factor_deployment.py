"""因子部署裁决框架测试（research_loop/factor_deployment.py）。

全部使用合成面板，快速、无外部数据依赖。覆盖：多空收益构造与单调性、方向对齐、
收益矩阵形状、PBO/DSR 取值域、最优因子识别、显性 alpha 低 PBO、composite、
报告文本、边界条件。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.backtest_mvp.research_loop.factor_deployment import (
    DeploymentVerdict,
    build_returns_matrix,
    evaluate_deployment,
    factor_long_short_returns,
    _single_period_sharpe,
)


# ─────────────────────────── 合成面板 ───────────────────────────

def _panel_index(n_dates=80, n_sym=20):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_sym)]
    return pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])


def _alpha_factor_and_returns(n_dates=80, n_sym=20, alpha=0.02, noise=0.01, seed=0):
    """构造一个真有 alpha 的因子：前视收益 = alpha * 截面标准化因子 + 噪声。"""
    rng = np.random.RandomState(seed)
    idx = _panel_index(n_dates, n_sym)
    sym_code = pd.Series(idx.get_level_values("symbol")).factorize()[0]
    date_code = pd.Series(idx.get_level_values("date")).factorize()[0]
    # 因子随日期缓变，避免完全恒定
    f = (sym_code - sym_code.mean()) / (sym_code.std() + 1e-9)
    f = f + rng.normal(0, 0.05, len(idx))
    factor = pd.Series(f, index=idx)
    fwd = alpha * f + rng.normal(0, noise, len(idx))
    fwd_returns = pd.Series(fwd, index=idx)
    return factor, fwd_returns


def _noise_factor(fwd_returns, seed=1):
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(0, 1, len(fwd_returns)), index=fwd_returns.index)


# ─────────────────────────── 多空收益构造 ───────────────────────────

def test_ls_returns_positive_for_real_alpha():
    factor, fwd = _alpha_factor_and_returns(alpha=0.05, noise=0.005)
    ls = factor_long_short_returns(factor, fwd, quantile=0.3)
    assert len(ls) > 0
    assert ls.mean() > 0, "真 alpha 因子的多空收益均值应为正"


def test_ls_returns_sign_flips_leg():
    factor, fwd = _alpha_factor_and_returns(alpha=0.05, noise=0.005)
    ls_pos = factor_long_short_returns(factor, fwd, sign=1.0)
    ls_neg = factor_long_short_returns(factor, fwd, sign=-1.0)
    # 反向后多空收益应近似取负
    assert np.sign(ls_pos.mean()) == -np.sign(ls_neg.mean())
    assert ls_pos.mean() == pytest.approx(-ls_neg.mean(), rel=1e-6)


def test_ls_returns_index_is_dates():
    factor, fwd = _alpha_factor_and_returns()
    ls = factor_long_short_returns(factor, fwd)
    # 索引应是唯一日期（非 MultiIndex）
    assert not isinstance(ls.index, pd.MultiIndex)
    assert ls.index.is_monotonic_increasing


def test_ls_returns_resample_step_subsamples():
    factor, fwd = _alpha_factor_and_returns(n_dates=60)
    full = factor_long_short_returns(factor, fwd, resample_step=1)
    stepped = factor_long_short_returns(factor, fwd, resample_step=5)
    assert 0 < len(stepped) < len(full)
    assert len(stepped) == len(range(0, len(full), 5))


def test_ls_returns_skips_thin_dates():
    # 每日只有 3 个标的，min_names=6 → 全跳过 → 空序列
    factor, fwd = _alpha_factor_and_returns(n_dates=20, n_sym=3)
    ls = factor_long_short_returns(factor, fwd, min_names=6)
    assert len(ls) == 0


def test_ls_returns_invalid_quantile():
    factor, fwd = _alpha_factor_and_returns()
    with pytest.raises(ValueError):
        factor_long_short_returns(factor, fwd, quantile=0.7)


# ─────────────────────────── 收益矩阵 ───────────────────────────

def test_returns_matrix_shape_and_columns():
    factor, fwd = _alpha_factor_and_returns()
    fvals = {"alpha": factor, "noise": _noise_factor(fwd)}
    mat, sharpes, signs = build_returns_matrix(fvals, fwd)
    assert list(mat.columns) == ["alpha", "noise"] or set(mat.columns) <= {"alpha", "noise"}
    assert mat.shape[0] > 0
    assert set(sharpes) == set(mat.columns)
    assert set(signs) == {"alpha", "noise"}


def test_returns_matrix_sign_alignment_positive_ic():
    # 负 IC 因子应被翻正 → sign=-1
    factor, fwd = _alpha_factor_and_returns(alpha=0.05, noise=0.005)
    neg_factor = -factor
    fvals = {"neg": neg_factor}
    mat, sharpes, signs = build_returns_matrix(fvals, fwd)
    assert signs["neg"] == -1.0
    # 翻正后多空 Sharpe 应为正
    assert sharpes["neg"] > 0


def test_returns_matrix_drops_thin_factor():
    factor, fwd = _alpha_factor_and_returns(n_dates=15)
    fvals = {"a": factor}
    mat, sharpes, signs = build_returns_matrix(fvals, fwd, min_periods=100)
    assert mat.empty
    assert sharpes == {}


def test_returns_matrix_handles_varying_warmup():
    # 两个因子 warmup 不同（前导 NaN 长度不同）：修复前逐因子 iloc[::step] 会错相位
    # 导致对齐后为空。此处 step>1 仍应得到非空、对齐良好的矩阵。
    factor, fwd = _alpha_factor_and_returns(n_dates=120, n_sym=20)
    dates = factor.index.get_level_values("date").unique()
    f_short = factor.copy()
    f_long = factor.copy()
    # f_long 前 40 个日期设为 NaN（长 warmup）
    warmup = set(dates[:40])
    mask = f_long.index.get_level_values("date").isin(warmup)
    f_long[mask] = np.nan
    mat, sharpes, signs = build_returns_matrix(
        {"short": f_short, "long": f_long}, fwd, resample_step=5, min_coverage=0.3
    )
    assert not mat.empty, "warmup 错位不应导致空矩阵"
    assert set(mat.columns) == {"short", "long"}
    assert mat.notna().all().all()


def test_min_coverage_drops_sparse_factor():
    factor, fwd = _alpha_factor_and_returns(n_dates=120, n_sym=20)
    dates = factor.index.get_level_values("date").unique()
    dense = factor.copy()
    sparse = factor.copy()
    # sparse 只保留最后 20% 日期 → 覆盖率低
    keep = set(dates[-24:])
    mask = ~sparse.index.get_level_values("date").isin(keep)
    sparse[mask] = np.nan
    mat, sharpes, signs = build_returns_matrix(
        {"dense": dense, "sparse": sparse}, fwd, min_coverage=0.6, resample_step=1
    )
    # sparse 被覆盖率过滤剔除，dense 保留且期数不被 sparse 拖垮
    assert "dense" in mat.columns
    assert "sparse" not in mat.columns
    assert mat.shape[0] > 24


# ─────────────────────────── 端到端裁决 ───────────────────────────

def test_evaluate_deployment_full_fields():
    factor, fwd = _alpha_factor_and_returns()
    fvals = {"alpha": factor, "noise1": _noise_factor(fwd, 1), "noise2": _noise_factor(fwd, 2)}
    v = evaluate_deployment(fvals, fwd, n_partitions=8)
    assert isinstance(v, DeploymentVerdict)
    assert v.n_factors == 3
    assert v.n_periods > 0
    assert 0.0 <= v.dsr <= 1.0
    assert 0.0 <= v.psr_best_vs_zero <= 1.0
    if v.pbo is not None:
        assert 0.0 <= v.pbo <= 1.0
    assert 0.0 <= v.composite_psr <= 1.0
    assert not v.per_factor.empty
    assert set(["factor", "ic", "sharpe", "selected"]).issubset(v.per_factor.columns)


def test_evaluate_best_factor_is_real_alpha():
    factor, fwd = _alpha_factor_and_returns(alpha=0.06, noise=0.004)
    fvals = {
        "alpha": factor,
        "noise1": _noise_factor(fwd, 11),
        "noise2": _noise_factor(fwd, 22),
        "noise3": _noise_factor(fwd, 33),
    }
    v = evaluate_deployment(fvals, fwd, n_partitions=8)
    assert v.best_factor == "alpha"
    assert v.best_sharpe > 0


def test_evaluate_dominant_alpha_low_pbo():
    # 一个强 alpha + 多个噪声 → IS 最优在 OOS 不退化 → PBO 应低
    factor, fwd = _alpha_factor_and_returns(alpha=0.06, noise=0.004, n_dates=120)
    fvals = {"alpha": factor}
    for i in range(6):
        fvals[f"noise{i}"] = _noise_factor(fwd, 100 + i)
    v = evaluate_deployment(fvals, fwd, n_partitions=10)
    assert v.pbo is not None
    assert v.pbo <= 0.35, f"显性 alpha 的 PBO 应偏低，实际 {v.pbo}"


def test_evaluate_composite_sharpe_finite():
    factor, fwd = _alpha_factor_and_returns()
    fvals = {"alpha": factor, "noise": _noise_factor(fwd)}
    v = evaluate_deployment(fvals, fwd, n_partitions=8)
    assert np.isfinite(v.composite_sharpe)
    assert 0.0 <= v.composite_psr <= 1.0


def test_evaluate_report_text_markers():
    factor, fwd = _alpha_factor_and_returns()
    fvals = {"alpha": factor, "noise": _noise_factor(fwd)}
    v = evaluate_deployment(fvals, fwd, n_partitions=8)
    for marker in ["因子部署裁决", "DSR", "Composite", "部署裁决"]:
        assert marker in v.report_text


def test_evaluate_empty_input():
    v = evaluate_deployment({}, pd.Series(dtype=float))
    assert v.n_factors == 0
    assert v.best_factor == ""
    assert v.pbo is None
    assert "无因子输入" in v.report_text


def test_evaluate_pbo_skipped_when_too_short():
    # 少量日期 → PBO 无法切块 → None，但 DSR 仍可算
    factor, fwd = _alpha_factor_and_returns(n_dates=12)
    fvals = {"a": factor, "b": _noise_factor(fwd)}
    v = evaluate_deployment(fvals, fwd, n_partitions=16)
    assert v.n_periods >= 1
    assert 0.0 <= v.dsr <= 1.0  # DSR 不依赖块切分


def test_to_dict_roundtrip_keys():
    factor, fwd = _alpha_factor_and_returns()
    fvals = {"alpha": factor, "noise": _noise_factor(fwd)}
    v = evaluate_deployment(fvals, fwd, n_partitions=8)
    d = v.to_dict()
    for key in ["pbo", "dsr", "best_factor", "composite_sharpe", "per_factor", "selected"]:
        assert key in d
    assert isinstance(d["per_factor"], list)


def test_single_period_sharpe_zero_variance():
    assert _single_period_sharpe(pd.Series([0.01, 0.01, 0.01])) == 0.0
    assert _single_period_sharpe(pd.Series([0.5])) == 0.0
    assert _single_period_sharpe(pd.Series([0.1, 0.2, 0.15, 0.05])) != 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
