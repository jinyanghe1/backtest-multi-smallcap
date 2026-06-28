#!/usr/bin/env python3
"""
Phase 2: Vectorized backtest — Fast loop with numpy optimization

Key optimizations over Phase 1 (engine_fast.py):
1. Pre-compute stock indices for all dates (avoid string lookups in loop)
2. Vectorized period return calculation (numpy matrix mean + cumprod)
3. Vectorized stop-loss check
4. Only pandas for ranking/sorting; everything else is numpy

Date: 2026-06-28
"""

import sys, numpy as np, pandas as pd
from typing import Dict, List, Optional, Callable

sys.path.insert(0, '/Users/hejinyang/thinking_and_learning_with_AI')
from tools.backtest_mvp.engine import (
    CrossSectionalEngine, BacktestResult,
    compute_monthly_ic_heatmap, _compute_max_drawdown_recovery_time,
    _compute_rolling_sharpe, _compute_turnover_attribution,
)
from tools.backtest_mvp.engine_precompute import EngineDataPrecomputation


class CrossSectionalEngineV2(CrossSectionalEngine):
    """Phase 2: Vectorized backtest loop"""

    def __init__(self, factor_panel, return_panel, initial_capital=1.0,
                 n_stocks=30, rebalance_freq='M', commission=0.0008,
                 slippage=0.002, price_limit_stocks=True):
        super().__init__(
            factor_panel=factor_panel, return_panel=return_panel,
            initial_capital=initial_capital, n_stocks=n_stocks,
            rebalance_freq=rebalance_freq, commission=commission,
            slippage=slippage, price_limit_stocks=price_limit_stocks,
        )
        # Phase 1 data precomputation
        self._pre = EngineDataPrecomputation(factor_panel, return_panel)
        self.rebalance_date_indices = [
            self._pre.date_to_idx[d] for d in self.rebalance_dates
        ]

    def run(self, universe_filter=None, ranking_factor="mcap", ascending=True,
            composite_factors=None, stop_loss=None, trailing_stop=None,
            take_profit_stocks=False, take_profit_threshold=0.50,
            ranking_fn=None, factor_weights=None, neutralize=False, neutralize_strength=0.5):

        equity = self.initial_capital
        equity_curve = [equity]
        equity_dates = [self.dates[0]]
        monthly_returns_log = []
        positions_log = {}
        turnover_log = []
        _stop_triggered = False
        _stop_date = None
        _peak_equity = self.initial_capital

        n_rebals = len(self.rebalance_dates)
        reb_indices = self.rebalance_date_indices

        for i in range(n_rebals - 1):
            rebal_idx = reb_indices[i]
            next_idx = reb_indices[i + 1]
            rebal_date = self.rebalance_dates[i]
            month_start_equity = equity

            # --- Fast factor snapshot ---
            snapshot = self._pre.get_factor_snapshot_fast(rebal_idx)
            if len(snapshot) == 0:
                continue

            # Drop stocks with NaN for the ranking factor
            if ranking_factor in snapshot.columns:
                snapshot = snapshot.dropna(subset=[ranking_factor])
            if len(snapshot) < self.n_stocks * 2:
                continue

            available_stocks = list(snapshot.index)

            # --- Universe filter ---
            if universe_filter is not None:
                selected = universe_filter(snapshot, self.dates, i)
            else:
                selected = available_stocks
            if len(selected) < self.n_stocks:
                continue

            # Price limit filter
            if self.price_limit_stocks and 'is_limit_up' in snapshot.columns:
                limit_up = set(snapshot[snapshot['is_limit_up'] == True].index)
                selected = [s for s in selected if s not in limit_up]

            # --- Neutralization ---
            if neutralize and ranking_factor in snapshot.columns:
                snapshot_neut = snapshot.copy()
                snapshot_neut[ranking_factor] = self._neutralize_snapshot(
                    snapshot, ranking_factor, strength=neutralize_strength
                )
                snapshot = snapshot_neut

            # --- Ranking ---
            max_pick = self.n_stocks * 2 if self.price_limit_stocks else self.n_stocks
            if ranking_fn is not None:
                scores = ranking_fn(snapshot.loc[selected])
                valid = scores.dropna()
                picked = valid.nlargest(self.n_stocks).index.tolist() if len(valid) > 0 else selected[:self.n_stocks]
            elif composite_factors is not None:
                factor_names = [f for f, _ in composite_factors]
                available = [f for f in factor_names if f in snapshot.columns]
                if len(available) > 0:
                    sub = snapshot.loc[selected][available].copy()
                    for col in available:
                        m, s = sub[col].mean(), sub[col].std()
                        sub[col] = (sub[col] - m) / s if s and s > 0 else 0
                    for f, asc in composite_factors:
                        if f in sub.columns and not asc:
                            sub[f] = -sub[f]
                    if factor_weights:
                        w_sum = pd.Series(0.0, index=sub.index)
                        total_w = 0.0
                        for f, asc in composite_factors:
                            if f in sub.columns:
                                w = factor_weights.get(f, 1.0 / len(composite_factors))
                                w_sum += sub[f] * w
                                total_w += w
                        composite = w_sum / total_w if total_w > 0 else sub.sum(axis=1, skipna=True)
                    else:
                        composite = sub.sum(axis=1, skipna=True)
                    ranked = composite.nsmallest(max_pick)
                    picked = ranked.index[:self.n_stocks].tolist()
                else:
                    picked = selected[:self.n_stocks]
            elif ranking_factor in snapshot.columns:
                valid = snapshot.loc[selected][ranking_factor].dropna()
                if ascending:
                    ranked = valid.nsmallest(max_pick)
                else:
                    ranked = valid.nlargest(max_pick)
                picked = ranked.index[:self.n_stocks].tolist()
            else:
                picked = selected[:self.n_stocks]

            if len(picked) == 0:
                continue

            weight = 1.0 / len(picked)

            # --- Turnover ---
            replaced_frac = 0.0
            if i > 0:
                prev_key = pd.Timestamp(self.rebalance_dates[i - 1])
                if prev_key in positions_log:
                    prev_picked = set(positions_log[prev_key].keys())
                    new_picked = set(picked)
                    replaced_frac = len(new_picked - prev_picked) / max(len(picked), 1)
                    turnover_log.append(replaced_frac)

            # --- Phase 2: Vectorized period returns ---
            picked_idx = self._pre.stock_list_to_indices(picked)
            picked_idx_arr = np.array(picked_idx)

            # Extract period returns (rebal_idx+1 to next_idx)
            period_rets = self._pre.returns_2d[rebal_idx + 1:next_idx + 1, :]
            period_rets = period_rets[:, picked_idx_arr]  # (days, n_picked)
            period_days = period_rets.shape[0]

            if period_days == 0:
                continue

            # Daily portfolio returns (equal weight)
            daily_rets = period_rets.mean(axis=1)

            # Deduct slippage + commission on first day
            daily_rets[0] -= self.slippage * replaced_frac + self.commission

            # Cumulative equity multipliers
            equity_mults = np.cumprod(1.0 + daily_rets)
            equity_curve_period = month_start_equity * equity_mults

            # Vectorized stop-loss check
            stop_triggered = False
            stop_date = None
            final_equity = equity_curve_period[-1]
            final_peak = max(_peak_equity, equity_curve_period.max())

            if stop_loss is not None:
                threshold = self.initial_capital * (1 + stop_loss)
                breach = np.where(equity_curve_period <= threshold)[0]
                if len(breach) > 0:
                    idx = breach[0]
                    stop_triggered = True
                    stop_date = self._pre.dates[rebal_idx + 1 + idx]
                    final_equity = equity_curve_period[idx]
                    equity_mults = equity_mults[:idx + 1]
                    period_days = idx + 1

            if not stop_triggered and trailing_stop is not None:
                running_peak = np.maximum.accumulate(equity_curve_period)
                breach = np.where(equity_curve_period <= running_peak * (1 - trailing_stop))[0]
                if len(breach) > 0:
                    idx = breach[0]
                    stop_triggered = True
                    stop_date = self._pre.dates[rebal_idx + 1 + idx]
                    final_equity = equity_curve_period[idx]
                    final_peak = running_peak[idx]
                    equity_mults = equity_mults[:idx + 1]
                    period_days = idx + 1

            # Record equity curve
            for j in range(period_days):
                equity_curve.append(month_start_equity * equity_mults[j])
                equity_dates.append(self._pre.dates[rebal_idx + 1 + j])

            equity = final_equity
            _peak_equity = final_peak

            monthly_returns_log.append(equity / month_start_equity - 1)
            positions_log[rebal_date] = {s: weight for s in picked}

            if stop_triggered:
                _stop_triggered = True
                _stop_date = stop_date
                break

        # --- Finalize ---
        if not _stop_triggered:
            equity *= (1 - self.commission)

        curve_data = list(zip(equity_dates[1:], equity_curve[1:]))
        if len(curve_data) == 0:
            equity_series = pd.Series([self.initial_capital] * len(self.rebalance_dates),
                                     index=self.rebalance_dates)
        else:
            equity_series = pd.Series(
                [self.initial_capital] + [v for _, v in curve_data],
                index=[self.dates[0]] + [d for d, _ in curve_data]
            )
            equity_series = equity_series.sort_index()

        monthly_ret = pd.Series(monthly_returns_log)
        n_months = len(monthly_ret)
        n_years = n_months / 12

        if n_years < 0.25:
            annual_ret = 0.0
            annual_vol = 0.0
        else:
            total_ret = equity_series.iloc[-1] / equity_series.iloc[0]
            annual_ret = total_ret ** (1 / n_years) - 1
            annual_vol = monthly_ret.std() * np.sqrt(12)

        sharpe = (annual_ret - 0.03) / annual_vol if annual_vol > 0 else 0

        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
        win_rate = (monthly_ret > 0).sum() / n_months if n_months > 0 else 0
        avg_turnover = np.mean(turnover_log) if turnover_log else 0

        # IC
        ic_series = self._compute_ic_series(
            ranking_factor, ranking_fn, composite_factors, factor_weights
        )
        if len(ic_series) > 0 and ic_series.notna().sum() > 0:
            valid_ic = ic_series.dropna()
            ic_mean = float(valid_ic.mean()) if len(valid_ic) > 0 else 0.0
            ic_std = float(valid_ic.std()) if len(valid_ic) > 1 else 0.0
            ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        else:
            ic_mean = 0.0
            ic_ir = 0.0

        q_spread = self._compute_quantile_spread(ranking_factor, ranking_fn, composite_factors, factor_weights)
        ic_heatmap = compute_monthly_ic_heatmap(ic_series)
        recovery_days = _compute_max_drawdown_recovery_time(equity_series)
        rolling_sharpe_series = _compute_rolling_sharpe(monthly_ret, window=12)
        turnover_attr = _compute_turnover_attribution(turnover_log, monthly_ret)

        result = BacktestResult(
            equity_curve=equity_series,
            monthly_returns=monthly_ret,
            annual_return=round(annual_ret * 100, 2),
            annual_volatility=round(annual_vol * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd * 100, 2),
            calmar_ratio=round(calmar, 2),
            win_rate=round(win_rate * 100, 1),
            avg_turnover=round(avg_turnover * 100, 1),
            terminal_value=round(equity, 4),
            positions_log=pd.DataFrame(positions_log).fillna(0),
            monthly_turnover_log=turnover_log,
            ic_mean=round(ic_mean, 4),
            ic_ir=round(ic_ir, 4),
            ic_series=ic_series,
            quantile_spread=round(q_spread, 6),
            monthly_ic_heatmap=ic_heatmap,
            max_drawdown_recovery_time=recovery_days,
            rolling_sharpe=rolling_sharpe_series,
            turnover_attribution=turnover_attr,
            stop_triggered=_stop_triggered,
            stop_trigger_date=str(_stop_date.date()) if _stop_date else "",
        )

        if n_months < 3:
            print(f"  ⚠️ [robust_start] 回测仅运行 {n_months} 个月, "
                  f"不足以评估策略长期表现。可能是在熊市中提前止损。")
        elif len(monthly_ret) >= 2:
            early_rets = monthly_ret[:min(3, len(monthly_ret))]
            if early_rets.mean() < -0.02:
                print(f"  ⚠️ [robust_start] 前 {len(early_rets)} 个月月均收益 "
                      f"{early_rets.mean()*100:.1f}% (连续亏损), "
                      f"回测起点可能处于熊市阶段。结果可能低估策略长期表现。")

        return result
