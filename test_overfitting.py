"""严谨过拟合诊断模块测试 (roadmap UA3 / research_loop/overfitting.py)。

对 PSR / E[max] / DSR / PBO 的**解析性质**做验证（合成数据，确定性，纯 stdlib+
numpy/scipy，无 engine / 实盘依赖）。
"""

import sys
import warnings
from math import comb
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.backtest_mvp.research_loop.overfitting import (
    sharpe_moments,
    probabilistic_sharpe_ratio,
    expected_maximum_sharpe,
    deflated_sharpe_ratio,
    deflated_sharpe_from_trials,
    probability_of_backtest_overfitting,
    overfitting_report,
)


# ─────────────────────────── PSR ───────────────────────────

def test_psr_half_at_benchmark():
    """观测 Sharpe == 基准 → PSR = 0.5。"""
    assert probabilistic_sharpe_ratio(0.3, 0.3, 120) == pytest.approx(0.5, abs=1e-9)


def test_psr_direction_around_benchmark():
    """SR>基准 → PSR>0.5；SR<基准 → PSR<0.5。"""
    assert probabilistic_sharpe_ratio(0.4, 0.2, 120) > 0.5
    assert probabilistic_sharpe_ratio(0.1, 0.2, 120) < 0.5


def test_psr_increases_with_n_obs():
    """样本越长, 超过基准的把握越大 (SR>基准时 PSR 单调↑)。"""
    lo = probabilistic_sharpe_ratio(0.3, 0.0, 24)
    hi = probabilistic_sharpe_ratio(0.3, 0.0, 240)
    assert hi > lo
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_psr_positive_skew_raises_negative_lowers():
    """正偏度抬高 PSR, 负偏度压低 (SR>0 时, 见标准误分母)。"""
    base = probabilistic_sharpe_ratio(0.3, 0.0, 120, skew=0.0)
    pos = probabilistic_sharpe_ratio(0.3, 0.0, 120, skew=0.8)
    neg = probabilistic_sharpe_ratio(0.3, 0.0, 120, skew=-0.8)
    assert pos > base > neg


def test_psr_fat_tails_lower():
    """厚尾 (峰度>3) 降低 PSR。"""
    normal = probabilistic_sharpe_ratio(0.3, 0.0, 120, kurt=3.0)
    fat = probabilistic_sharpe_ratio(0.3, 0.0, 120, kurt=10.0)
    assert fat < normal


def test_psr_invalid_n_obs_raises():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.3, 0.0, 1)


# ─────────────────────────── E[max] ───────────────────────────

def test_expected_max_increases_with_trials():
    """试验越多, 运气最优的 Sharpe 期望越大。"""
    e10 = expected_maximum_sharpe(10, 0.02)
    e100 = expected_maximum_sharpe(100, 0.02)
    e1000 = expected_maximum_sharpe(1000, 0.02)
    assert 0 < e10 < e100 < e1000


def test_expected_max_zero_when_no_variance():
    """跨试验方差为 0 → 无膨胀 → E[max]=0。"""
    assert expected_maximum_sharpe(500, 0.0) == 0.0


def test_expected_max_single_trial_is_zero():
    """N=1 → 无多重检验 → E[max]=0 (护栏, 不触 Φ⁻¹(0)=−∞)。"""
    assert expected_maximum_sharpe(1, 0.05) == 0.0


def test_expected_max_invalid_trials_raises():
    with pytest.raises(ValueError):
        expected_maximum_sharpe(0, 0.02)


# ─────────────────────────── DSR ───────────────────────────

def test_dsr_single_trial_equals_psr_vs_zero():
    """N=1 时基准=0, DSR 退化为 PSR(vs 0), 不做折扣。"""
    d = deflated_sharpe_ratio(0.3, 0.02, n_trials=1, n_obs=120)
    p0 = probabilistic_sharpe_ratio(0.3, 0.0, 120)
    assert d == pytest.approx(p0, abs=1e-12)


def test_dsr_deflated_below_psr0_with_many_trials():
    """N 大 + 跨试验方差>0 → DSR 明显低于 PSR(vs 0)（去膨胀生效）。"""
    d = deflated_sharpe_ratio(0.3, 0.02, n_trials=100, n_obs=120)
    p0 = probabilistic_sharpe_ratio(0.3, 0.0, 120)
    assert d < p0
    assert 0.0 <= d <= 1.0


def test_dsr_genuine_beats_best_of_noise():
    """真实强单策略 (N 少、方差低) 的 DSR 高于'噪声里挑最优' (N 多、方差高)。"""
    genuine = deflated_sharpe_from_trials([0.32, 0.30, 0.31], n_obs=120)
    noise = deflated_sharpe_from_trials(
        list(np.linspace(-0.25, 0.32, 60)), n_obs=120
    )
    assert genuine["dsr"] > noise["dsr"]


def test_deflated_sharpe_from_trials_fields():
    """便捷入口返回字段完整, best=max, N=len。"""
    out = deflated_sharpe_from_trials([0.1, 0.25, -0.05, 0.2], n_obs=100)
    assert out["best_sharpe"] == pytest.approx(0.25)
    assert out["n_trials"] == 4
    for k in ("dsr", "expected_max_sharpe", "sr_variance", "psr_vs_zero"):
        assert k in out
    assert 0.0 <= out["dsr"] <= 1.0


# ─────────────────────────── PBO / CSCV ───────────────────────────

def _rng(seed=11):
    return np.random.RandomState(seed)


def test_pbo_dominant_strategy_low():
    """存在一列真正占优 → IS 最优在 OOS 仍好 → PBO 低 (≈0)。"""
    T, N = 120, 20
    M = _rng().normal(0, 0.05, size=(T, N))
    M[:, 0] += 0.02  # 稳定 alpha
    res = probability_of_backtest_overfitting(M, n_partitions=10)
    assert res["pbo"] < 0.2


def test_pbo_pure_noise_near_half():
    """全噪声 → IS 最优纯属偶然 → PBO 期望 = 0.5。

    单次抽样的 PBO 方差大 (std≈0.2), 故对 16 个独立噪声矩阵取均值 (确定性 seed),
    验证 **期望** 收敛到 0.5 附近, 而非依赖单次运气。
    """
    vals = [
        probability_of_backtest_overfitting(
            np.random.RandomState(s).normal(0, 0.05, size=(160, 24)),
            n_partitions=10,
        )["pbo"]
        for s in range(16)
    ]
    assert 0.38 < float(np.mean(vals)) < 0.62


def test_pbo_constructed_overfit_high():
    """构造过拟合: 第 j 列只在第 j 块爆发 → IS 最优在 OOS 必差 → PBO 高。"""
    S, N, block = 8, 8, 15
    T = S * block
    M = np.full((T, N), -0.001)
    for j in range(N):
        M[j * block:(j + 1) * block, j] = 0.05
    res = probability_of_backtest_overfitting(M, n_partitions=S)
    assert res["pbo"] > 0.8
    # 过拟合下 OOS 最优秩中位数应偏低
    assert res["median_oos_rank"] < 0.5


def test_pbo_n_combinations_is_C_S_half():
    """组合数 = C(S, S/2)。"""
    M = _rng(5).normal(0, 0.05, size=(120, 10))
    res = probability_of_backtest_overfitting(M, n_partitions=8)
    assert res["n_combinations"] == comb(8, 4)


def test_pbo_custom_metric_fn():
    """自定义评价函数 (列均值) 可用, 返回结构完整。"""
    M = _rng(6).normal(0.001, 0.05, size=(120, 12))
    res = probability_of_backtest_overfitting(
        M, n_partitions=8, metric_fn=lambda x: x.mean(axis=0)
    )
    assert 0.0 <= res["pbo"] <= 1.0
    assert len(res["logits"]) == comb(8, 4)


def test_pbo_odd_partitions_raises():
    M = _rng().normal(0, 0.05, size=(120, 10))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(M, n_partitions=7)


def test_pbo_too_few_strategies_raises():
    M = _rng().normal(0, 0.05, size=(120, 1))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(M, n_partitions=8)


def test_pbo_too_few_rows_raises():
    M = _rng().normal(0, 0.05, size=(6, 10))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(M, n_partitions=8)


def test_pbo_no_warnings_emitted():
    """低方差 IS 得分下也不应触发 numpy RankWarning。"""
    S, N, block = 8, 8, 15
    M = np.full((S * block, N), -0.001)
    for j in range(N):
        M[j * block:(j + 1) * block, j] = 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probability_of_backtest_overfitting(M, n_partitions=S)


# ─────────────────────── sharpe_moments / report ───────────────────────

def test_sharpe_moments_basic():
    """已知序列: mean/std/sharpe 正确, 近正态偏度≈0。"""
    r = _rng(1).normal(0.01, 0.05, size=500)
    m = sharpe_moments(r)
    assert m["n_obs"] == 500
    assert m["sharpe"] == pytest.approx(m["mean"] / m["std"], rel=1e-9)
    assert abs(m["skew"]) < 0.5
    assert 2.0 < m["kurtosis"] < 4.0


def test_sharpe_moments_too_short_raises():
    with pytest.raises(ValueError):
        sharpe_moments([0.01])


def test_overfitting_report_contains_verdict_and_metrics():
    """报告字符串含 DSR / PBO / 裁决。"""
    txt = overfitting_report(
        observed_sr=0.35, n_trials=51, n_obs=120,
        sr_variance_across_trials=0.012, pbo=0.42,
    )
    assert "DSR" in txt and "裁决" in txt and "PBO" in txt


def test_report_verdict_flips_with_significance():
    """高 DSR → 稳健裁决; 低 DSR → 存疑裁决。"""
    strong = overfitting_report(
        observed_sr=0.6, n_trials=1, n_obs=240, sr_variance_across_trials=0.0
    )
    weak = overfitting_report(
        observed_sr=0.15, n_trials=500, n_obs=60, sr_variance_across_trials=0.05
    )
    assert "稳健" in strong
    assert "存疑" in weak


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
