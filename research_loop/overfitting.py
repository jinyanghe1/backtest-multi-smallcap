"""严谨的多重检验 / 回测过拟合诊断（roadmap UA3）。

本模块提供 **严谨版** 的 Probabilistic / Deflated Sharpe Ratio 与
**Probability of Backtest Overfitting (PBO)**，用于回答量化研究里最要命的问题：

    "在跑了 N 个因子/策略变体、只报告最优 Sharpe 之后，这个 Sharpe 还是真的吗？"

与同目录 `deflated_sharpe.py` 的关系
------------------------------------
`deflated_sharpe.py` 用的是 **粗糙 haircut 近似**
    DSR ≈ SR − ln(N) / (4·√T)
它是一个便捷下界，但既不含偏度/峰度修正，也没有 López de Prado 的 E[max] 去膨胀
基准，significance 用了不成立的分布假设。本模块**不修改**它（保持向后兼容），而是
另起严谨实现，供需要"可发表级"诊断的场景使用。

方法与参考
----------
- Probabilistic Sharpe Ratio (PSR) / Deflated Sharpe Ratio (DSR):
  Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  Journal of Portfolio Management, 40(5), 94-107.
- Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
  Cross-Validation (CSCV):
  Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017).
  "The Probability of Backtest Overfitting." Journal of Computational
  Finance, 20(4), 39-69.

约定（重要）
------------
PSR/DSR 使用 **单期（非年化）** Sharpe 与 **期数 n_obs**。若你手上是年化 Sharpe，
请先除以 √(periods_per_year) 还原为单期 Sharpe，或用 `sharpe_moments()` 从收益率
序列直接求 (单期SR, 偏度, 峰度, 期数)。

纯 numpy/scipy，无项目内耦合，不触碰 engine / 实盘数据。
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import stats

# Euler–Mascheroni 常数（E[max] 极值近似用）
_EULER_GAMMA = 0.5772156649015329
# 单期 Sharpe 估计量标准误的分母下界（防偏度/峰度极端时非正）
_VAR_FLOOR = 1e-12


# ───────────────────────────── 收益率矩 ─────────────────────────────

def sharpe_moments(
    returns: Sequence[float],
    *,
    ddof: int = 1,
) -> Dict[str, float]:
    """从收益率序列求单期 Sharpe 及高阶矩，供 PSR/DSR 使用。

    Args:
        returns: 单期收益率序列（月/日均可，PSR 用哪种就传哪种）。
        ddof: 标准差自由度修正（默认 1 = 样本标准差）。

    Returns:
        {"sharpe": 单期SR, "skew": 偏度, "kurtosis": 峰度(非超额, 正态=3),
         "n_obs": 期数, "mean": 均值, "std": 标准差}
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 2:
        raise ValueError("returns must contain >= 2 finite observations")
    mean = float(r.mean())
    std = float(r.std(ddof=ddof))
    sharpe = mean / std if std > 0 else 0.0
    # scipy.skew/kurtosis: bias=True 给总体矩; fisher=False -> 峰度正态=3
    skew = float(stats.skew(r, bias=True)) if std > 0 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False, bias=True)) if std > 0 else 3.0
    return {
        "sharpe": sharpe,
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": n,
        "mean": mean,
        "std": std,
    }


def _sharpe_se_denominator(sr: float, skew: float, kurt: float) -> float:
    """单期 Sharpe 估计量标准误的分母 √(1 − γ3·SR + (γ4−1)/4·SR²)。

    源自 Mertens / Lo 的 Sharpe 估计量渐近方差（含非正态修正）。对正态
    (skew=0, kurt=3) 退化为 √(1 + SR²/2)。用下界防极端偏度导致非正。
    """
    var = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)
    return math.sqrt(max(var, _VAR_FLOOR))


# ─────────────────────── Probabilistic Sharpe Ratio ───────────────────────

def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR)。

    PSR = Φ[ (SR̂ − SR*)·√(n−1) / √(1 − γ3·SR̂ + (γ4−1)/4·SR̂²) ]

    含义：在给定样本长度与收益率形状（偏度/峰度）下，真实 Sharpe 超过基准
    SR* 的概率。返回 [0, 1]。

    Args:
        observed_sr: 观测到的 **单期** Sharpe（非年化）。
        benchmark_sr: 基准 **单期** Sharpe（常用 0；DSR 会填入 E[max]）。
        n_obs: 样本期数（观测数）。
        skew: 收益率偏度（正态=0）。
        kurt: 收益率峰度（非超额，正态=3）。

    Returns:
        PSR ∈ [0, 1]。
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    denom = _sharpe_se_denominator(observed_sr, skew, kurt)
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


# ─────────────────────── E[max Sharpe] 去膨胀基准 ───────────────────────

def expected_maximum_sharpe(n_trials: int, sr_variance: float) -> float:
    """N 次独立试验下 Sharpe 最大值的期望（López de Prado 极值近似）。

    E[max_N SR] ≈ √Var · [ (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

    其中 γ = Euler–Mascheroni，Z⁻¹ = 标准正态分位数，假设各试验 Sharpe 在
    原假设下 ~ N(0, Var)。这是"运气最好的那次"应有的 Sharpe，作为 DSR 的去膨胀基准。

    Args:
        n_trials: 试验次数 N（跑过的策略/参数组合数）。
        sr_variance: 跨试验 Sharpe 估计的方差 Var(SR_trials)（单期口径）。

    Returns:
        E[max] Sharpe（单期）。N<=1 或 Var<=0 时返回 0.0（无膨胀可去）。
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1 or sr_variance <= 0:
        return 0.0
    sigma = math.sqrt(sr_variance)
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1.0 - _EULER_GAMMA) * q1 + _EULER_GAMMA * q2))


# ─────────────────────────── Deflated Sharpe Ratio ───────────────────────────

def deflated_sharpe_ratio(
    observed_sr: float,
    sr_variance_across_trials: float,
    n_trials: int,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR) — 严谨版。

    DSR = PSR(SR* = E[max_N SR])
        = Φ[ (SR̂ − E[max])·√(n−1) / √(1 − γ3·SR̂ + (γ4−1)/4·SR̂²) ]

    即"观测到的最优 Sharpe 在去膨胀基准之上仍显著"的概率。DSR→1 = 扣除多重检验
    选择偏差后依然可信；DSR→0 = 大概率是 N 次里运气最好的那次。

    Args:
        observed_sr: 观测到的最优 **单期** Sharpe。
        sr_variance_across_trials: 跨所有试验的 Sharpe 方差（单期口径）。
        n_trials: 试验次数 N。
        n_obs: 样本期数。
        skew / kurt: 最优策略收益率的偏度 / 峰度（非超额峰度，正态=3）。

    Returns:
        DSR ∈ [0, 1]。
    """
    benchmark = expected_maximum_sharpe(n_trials, sr_variance_across_trials)
    return probabilistic_sharpe_ratio(
        observed_sr, benchmark, n_obs, skew=skew, kurt=kurt
    )


def deflated_sharpe_from_trials(
    trial_sharpes: Sequence[float],
    n_obs: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    best_sr: Optional[float] = None,
) -> Dict[str, float]:
    """便捷入口：给定一组试验的单期 Sharpe，直接算 DSR。

    自动取 best = max(trial_sharpes)、N = len、Var = 跨试验方差。

    Args:
        trial_sharpes: 所有试验的 **单期** Sharpe 列表。
        n_obs: 样本期数。
        skew / kurt: 最优策略收益率形状。
        best_sr: 覆盖"最优 Sharpe"（默认取列表最大值）。

    Returns:
        {"dsr", "best_sharpe", "n_trials", "expected_max_sharpe",
         "sr_variance", "psr_vs_zero"}
    """
    arr = np.asarray(list(trial_sharpes), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        raise ValueError("trial_sharpes must contain >= 1 finite value")
    n_trials = int(arr.size)
    best = float(arr.max()) if best_sr is None else float(best_sr)
    var = float(arr.var(ddof=1)) if arr.size >= 2 else 0.0
    emax = expected_maximum_sharpe(n_trials, var)
    dsr = probabilistic_sharpe_ratio(best, emax, n_obs, skew=skew, kurt=kurt)
    psr0 = probabilistic_sharpe_ratio(best, 0.0, n_obs, skew=skew, kurt=kurt)
    return {
        "dsr": dsr,
        "best_sharpe": best,
        "n_trials": n_trials,
        "expected_max_sharpe": emax,
        "sr_variance": var,
        "psr_vs_zero": psr0,
    }


# ──────────────── Probability of Backtest Overfitting (PBO / CSCV) ────────────────

def _column_sharpe(matrix: np.ndarray) -> np.ndarray:
    """每列（策略）的单期 Sharpe；零方差列记 0。"""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    out = np.zeros_like(mean, dtype=float)
    nz = std > 0
    out[nz] = mean[nz] / std[nz]
    return out


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    *,
    n_partitions: int = 16,
    metric_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, object]:
    """通过 CSCV 估计回测过拟合概率 (PBO)。

    输入 T×N 收益率矩阵（T 期 × N 个策略/因子）。把时间轴切成 S 个等长块，
    枚举所有 C(S, S/2) 种"训练块/测试块"对称划分；每种划分里取样本内(IS)表现
    最好的策略，看它在样本外(OOS)的**相对秩** ω，logit λ=ln(ω/(1−ω))。

        PBO = P(λ ≤ 0) = IS 最优策略在 OOS 落到中位数以下的频率

    PBO 高（→0.5+）说明"选出来的最优"多半是过拟合噪声。

    Args:
        returns_matrix: 形如 (T, N) 的 ndarray / 可转 ndarray；行=时间，列=策略。
        n_partitions: S，时间块数，必须为 **偶数** 且 ≥ 2（默认 16）。
        metric_fn: 子矩阵(2D)→每列得分(1D) 的评价函数，默认单期 Sharpe。

    Returns:
        {
          "pbo": float,                     # 过拟合概率 ∈ [0,1]
          "n_combinations": int,
          "logits": list[float],
          "performance_degradation_slope": float,  # OOS(best) ~ IS(best) 回归斜率
          "prob_oos_loss": float,           # IS 最优在 OOS 亏损(得分<0)的频率
          "median_oos_rank": float,         # IS 最优的 OOS 相对秩中位数
        }
    """
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be 2D (T periods x N strategies)")
    if metric_fn is None:
        metric_fn = _column_sharpe

    S = int(n_partitions)
    if S < 2 or S % 2 != 0:
        raise ValueError("n_partitions must be an even integer >= 2")

    T, N = m.shape
    if N < 2:
        raise ValueError("need >= 2 strategies (columns) to rank")
    if T < S:
        raise ValueError(f"need >= n_partitions ({S}) rows, got {T}")

    # 切成 S 个等长块（丢弃无法整除的尾部，保证块等长）
    block_len = T // S
    usable = block_len * S
    blocks = [m[i * block_len:(i + 1) * block_len] for i in range(S)]

    half = S // 2
    all_block_idx = set(range(S))
    logits: list[float] = []
    is_best_scores: list[float] = []
    oos_best_scores: list[float] = []
    oos_ranks: list[float] = []

    for is_idx in combinations(range(S), half):
        oos_idx = tuple(sorted(all_block_idx - set(is_idx)))
        is_mat = np.vstack([blocks[b] for b in is_idx])
        oos_mat = np.vstack([blocks[b] for b in oos_idx])

        is_perf = np.asarray(metric_fn(is_mat), dtype=float)
        oos_perf = np.asarray(metric_fn(oos_mat), dtype=float)

        n_star = int(np.nanargmax(is_perf))          # IS 最优策略
        oos_val = float(oos_perf[n_star])

        # OOS 相对秩 ∈ (0,1)：用平均秩防并列，rank/(N+1) 保证不触 0/1
        ranks = stats.rankdata(oos_perf, method="average")
        omega = float(ranks[n_star] / (N + 1.0))
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(omega / (1.0 - omega)))

        is_best_scores.append(float(is_perf[n_star]))
        oos_best_scores.append(oos_val)
        oos_ranks.append(omega)

    logits_arr = np.asarray(logits)
    pbo = float(np.mean(logits_arr <= 0.0))

    # 性能衰减：OOS(best) 对 IS(best) 的最小二乘斜率（过拟合→低/负斜率）
    isb = np.asarray(is_best_scores)
    oosb = np.asarray(oos_best_scores)
    var_is = float(isb.var())
    if isb.size >= 2 and var_is > 0:
        # 手算斜率 cov/var，避免 np.polyfit 在 IS 得分近乎恒定时的 RankWarning
        slope = float(np.cov(isb, oosb, ddof=0)[0, 1] / var_is)
    else:
        slope = 0.0

    return {
        "pbo": pbo,
        "n_combinations": len(logits),
        "logits": logits,
        "performance_degradation_slope": slope,
        "prob_oos_loss": float(np.mean(oosb < 0.0)),
        "median_oos_rank": float(np.median(oos_ranks)),
    }


# ─────────────────────────────── 报告 ───────────────────────────────

def overfitting_report(
    *,
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    sr_variance_across_trials: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    pbo: Optional[float] = None,
) -> str:
    """生成人类可读的过拟合诊断汇总（单期 Sharpe 口径）。"""
    emax = expected_maximum_sharpe(n_trials, sr_variance_across_trials)
    psr0 = probabilistic_sharpe_ratio(observed_sr, 0.0, n_obs, skew=skew, kurt=kurt)
    dsr = probabilistic_sharpe_ratio(observed_sr, emax, n_obs, skew=skew, kurt=kurt)

    if dsr >= 0.95:
        verdict = "稳健：扣除多重检验后仍高度显著"
    elif dsr >= 0.90:
        verdict = "尚可：去膨胀后边际显著，谨慎采纳"
    else:
        verdict = "存疑：大概率是多重检验的运气最优，勿直接部署"

    lines = [
        "═══ 过拟合 / 多重检验诊断 (roadmap UA3) ═══",
        f"  观测单期 Sharpe      : {observed_sr:.4f}",
        f"  样本期数 n_obs        : {n_obs}",
        f"  试验次数 N            : {n_trials}",
        f"  跨试验 Sharpe 方差    : {sr_variance_across_trials:.6f}",
        f"  偏度 / 峰度           : {skew:.3f} / {kurt:.3f}",
        f"  E[max] 去膨胀基准     : {emax:.4f}",
        f"  PSR (vs 0)           : {psr0:.4f}",
        f"  DSR (vs E[max])      : {dsr:.4f}",
    ]
    if pbo is not None:
        lines.append(f"  PBO (CSCV)           : {pbo:.4f}")
    lines.append(f"  裁决                 : {verdict}")
    return "\n".join(lines)


# ─────────────────────────── 离线自检 ───────────────────────────

def _synthetic_selfcheck() -> None:  # pragma: no cover - CLI 演示
    rng = np.random.RandomState(7)
    T, N = 120, 25
    # 一列真实占优 + 其余纯噪声
    noise = rng.normal(0, 0.05, size=(T, N))
    noise[:, 0] += 0.02  # 策略0 真有 alpha
    res = probability_of_backtest_overfitting(noise, n_partitions=10)
    print("[selfcheck] 单一占优策略 PBO =", round(res["pbo"], 3),
          "(期望≈0)  slope =", round(res["performance_degradation_slope"], 3))

    pure = rng.normal(0, 0.05, size=(T, N))  # 全噪声
    res2 = probability_of_backtest_overfitting(pure, n_partitions=10)
    print("[selfcheck] 全噪声 PBO =", round(res2["pbo"], 3), "(期望≈0.5)")

    trial_srs = list(rng.normal(0, 0.1, size=50)) + [0.35]  # 50 噪声 + 1 高
    d = deflated_sharpe_from_trials(trial_srs, n_obs=120)
    print("[selfcheck] DSR(best-of-51) =", round(d["dsr"], 3),
          " E[max] =", round(d["expected_max_sharpe"], 3),
          " PSR(vs0) =", round(d["psr_vs_zero"], 3))
    print(overfitting_report(
        observed_sr=0.35, n_trials=51, n_obs=120,
        sr_variance_across_trials=d["sr_variance"], pbo=res2["pbo"],
    ))


if __name__ == "__main__":  # pragma: no cover
    _synthetic_selfcheck()
