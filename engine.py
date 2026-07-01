"""
截面回测引擎 — Cross-Sectional Walk-Forward Backtest Engine
=====
设计目标: 支持 A 股微盘股截面策略 (月度排序→选 top N→持有至下次调仓)

核心假设 (你的约束条件):
  - 仅日线级别, 无日内数据
  - 月度调仓 (可配)
  - 有限标的 (≤1000 只)
  - 不考虑 T+1/涨跌停简化版 (可通过 price_limit 开关控制)

输入: 因子面板 DataFrame (date × stock, columns = [mcap, pb, mom20d, turnover, ...])
         + 收益率面板 DataFrame (date × stock, values = 日收益率)
输出: 权益曲线, 年化收益, 最大回撤, 夏普, 换手率, 逐月持仓记录
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple

from tools.backtest_mvp.factors.preprocessing import preprocess_pipeline


@dataclass
class BacktestResult:
    """回测输出"""
    equity_curve: pd.Series          # 逐日权益曲线
    monthly_returns: pd.Series       # 逐月收益率
    annual_return: float             # 年化收益率
    annual_volatility: float         # 年化波动率
    sharpe_ratio: float              # 年化夏普 (无风险利率假设为 3%)
    max_drawdown: float              # 最大回撤
    calmar_ratio: float              # 年化/最大回撤
    win_rate: float                  # 月度胜率
    avg_turnover: float              # 平均每月换手率 (调仓时替换的股票比例)
    terminal_value: float            # 期末权益 (初始=1)
    positions_log: pd.DataFrame      # 逐月持仓明细 (date × stock → weight)
    monthly_turnover_log: List[float] # 逐月换手率
    ic_mean: float = 0.0             # 平均信息系数 (Spearman rank IC)
    ic_ir: float = 0.0               # IC 信息比率 = ic_mean / ic_std
    ic_series: pd.Series = None      # 逐调仓日 IC 序列
    quantile_spread: float = 0.0     # Q5-Q1 return spread (top-bottom quantile)
    monthly_ic_heatmap: pd.DataFrame = None  # IC 按月热力图 (index=year, columns=month)
    max_drawdown_recovery_time: int = 0  # 最大回撤恢复天数 (谷底→创新高)
    rolling_sharpe: pd.Series = None      # 滚动夏普比率序列
    turnover_attribution: dict = None     # 换手率分解 {rebalance, price_drift, total}
    stop_triggered: bool = False    # 是否触发了组合止损
    stop_trigger_date: str = ""     # 止损触发日期 (YYYY-MM-DD)


def compute_monthly_ic_heatmap(ic_series: pd.Series) -> pd.DataFrame:
    """Pivot IC series into a year × month heatmap.

    Parameters
    ----------
    ic_series : pd.Series indexed by date (rebalance dates)

    Returns
    -------
    DataFrame with index=year, columns=1..12, values=mean IC for that month.
    NaN where no data exists for a given year-month.
    """
    if ic_series is None or ic_series.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(ic_series.index)
    df = pd.DataFrame({
        "year": idx.year,
        "month": idx.month,
        "ic": ic_series.values,
    })
    heatmap = df.groupby(["year", "month"])["ic"].mean().unstack(level="month")
    # Ensure columns are 1..12 even if some months are missing
    heatmap = heatmap.reindex(columns=range(1, 13))
    heatmap.columns.name = "month"
    heatmap.index.name = "year"
    return heatmap


def _compute_max_drawdown_recovery_time(equity_curve: pd.Series) -> int:
    """Days from max-drawdown trough to the date the curve first reaches a new high.

    Returns 0 if the curve never made a new high after the trough,
    or if there is no drawdown.
    """
    if equity_curve is None or equity_curve.empty:
        return 0
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    trough_idx = dd.idxmin()
    trough_val = equity_curve.loc[trough_idx]
    # Find first date after trough where equity >= pre-trough peak
    pre_trough_peak = peak.loc[trough_idx]
    after = equity_curve.loc[trough_idx:]
    new_high_dates = after[after >= pre_trough_peak].index
    if len(new_high_dates) == 0:
        return 0
    recovery_date = new_high_dates[0]
    # Count trading days (index entries) between trough and recovery
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        return int((recovery_date - trough_idx).days)
    return int(equity_curve.index.get_loc(recovery_date) - equity_curve.index.get_loc(trough_idx))


def _compute_rolling_sharpe(monthly_returns: pd.Series, window: int = 12) -> pd.Series:
    """Rolling annualized Sharpe ratio from monthly returns.

    Uses a rolling window of `window` months, annualizes by sqrt(12),
    and assumes 3% risk-free rate (annual).
    """
    if monthly_returns is None or monthly_returns.empty or len(monthly_returns) < 2:
        return pd.Series(dtype=float)
    rolling_mean = monthly_returns.rolling(window, min_periods=max(2, window // 2)).mean()
    rolling_std = monthly_returns.rolling(window, min_periods=max(2, window // 2)).std()
    annual_rf_monthly = (1 + 0.03) ** (1 / 12) - 1
    excess = rolling_mean - annual_rf_monthly
    annualized = excess / rolling_std.replace(0, np.nan) * np.sqrt(12)
    return annualized.dropna()


def _compute_turnover_attribution(
    turnover_log: list[float],
    monthly_returns: pd.Series,
) -> dict:
    """Decompose turnover into rebalancing-driven and price-drift components.

    - ``rebalance_turnover``: average turnover from explicit rebalancing
      (what the engine already tracks — fraction of portfolio replaced).
    - ``price_drift_turnover``: estimated passive turnover from price changes
      within each holding period, computed as the average absolute deviation
      of monthly returns from zero (proxy for weight drift).
    - ``total_turnover``: sum of both components.

    Returns a dict with float values (0–1 scale).
    """
    rebal_avg = float(np.mean(turnover_log)) if turnover_log else 0.0
    # Price drift proxy: average |monthly return| measures how much weights
    # would drift if no rebalancing occurred
    if monthly_returns is not None and len(monthly_returns) > 0:
        drift_proxy = float(monthly_returns.abs().mean()) / 2.0
    else:
        drift_proxy = 0.0
    return {
        "rebalance_turnover": rebal_avg,
        "price_drift_turnover": drift_proxy,
        "total_turnover": rebal_avg + drift_proxy,
    }


class CrossSectionalEngine:
    """
    截面回测引擎

    用法:
        engine = CrossSectionalEngine(
            factor_panel=factors,
            return_panel=returns,
            n_stocks=30,
            rebalance_freq='M',
        )
        result = engine.run(universe_filter, ranking_fn)
    """

    def __init__(
        self,
        factor_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
        initial_capital: float = 1.0,
        n_stocks: int = 30,
        rebalance_freq: str = 'M',
        commission: float = 0.0008,   # A股: 万2.5佣金 + 千1印花税 ≈ 0.125%, 留余量
        slippage: float = 0.002,      # 微盘股滑点 0.2%
        price_limit_stocks: bool = True,  # 是否过滤涨停/跌停
    ):
        self.factors = factor_panel
        self.returns = return_panel
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks
        self.commission = commission
        self.slippage = slippage
        self.price_limit_stocks = price_limit_stocks

        # 对齐因子和收益率的时间轴 — 统一为 pd.Timestamp
        self.dates = sorted(
            pd.Timestamp(d) for d in (
                set(factor_panel.index.get_level_values(0))
                & set(return_panel.index.get_level_values(0))
            )
        )
        self.stocks = sorted(set(factor_panel.index.get_level_values(1))
                           & set(return_panel.index.get_level_values(1)))

        # 生成调仓日
        all_dates = pd.DatetimeIndex(self.dates)
        if rebalance_freq == 'M':
            self.rebalance_dates = self._get_monthly_dates(all_dates)
        elif rebalance_freq == 'W':
            self.rebalance_dates = self._get_weekly_dates(all_dates)
        else:
            self.rebalance_dates = all_dates

    def _get_monthly_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """获取每月最后一个交易日作为调仓日"""
        df = pd.DataFrame({'date': dates})
        df['month'] = dates.to_period('M')
        return pd.DatetimeIndex(df.groupby('month')['date'].last().values)

    def _get_weekly_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """获取每周最后一个交易日"""
        df = pd.DataFrame({'date': dates})
        df['week'] = dates.to_period('W')
        return pd.DatetimeIndex(df.groupby('week')['date'].last().values)

    def _get_factor_snapshot(self, date) -> pd.DataFrame:
        """获取指定日期的因子横截面, 返回 index=stock_code 的 DataFrame"""
        try:
            ts = pd.Timestamp(date)
            f = self.factors.xs(ts, level=0, drop_level=True)
            return f
        except (KeyError, TypeError, AttributeError):
            return pd.DataFrame()

    def _get_daily_return(self, date, stocks: List) -> pd.Series:
        """获取指定日期的个股收益率"""
        ts = pd.Timestamp(date)
        try:
            r = self.returns.xs(ts, level=0, drop_level=True)
            r = r.reindex(stocks)
            return r['daily_return'].fillna(0)
        except (KeyError, AttributeError):
            return pd.Series(0.0, index=stocks)

    def _get_period_returns(self, start_date, end_date, stocks: List) -> pd.DataFrame:
        """获取持仓期间的日收益率矩阵 (dates × stocks)"""
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        dates_in_range = [d for d in self.dates if d > start_ts and d <= end_ts]
        if not dates_in_range:
            return pd.DataFrame(columns=stocks)
        try:
            r_slice = self.returns.loc[pd.IndexSlice[dates_in_range, stocks], 'daily_return']
        except KeyError:
            return pd.DataFrame(columns=stocks)
        if len(r_slice) == 0:
            return pd.DataFrame(columns=stocks)
        pivot = r_slice.unstack(level=1)
        return pivot.reindex(columns=stocks, fill_value=0.0)

    def _neutralize_snapshot(self, snapshot, factor_col, strength=0.5):
        """Neutralize a single factor column in the snapshot (size + industry)."""
        # Build controls from snapshot
        controls = pd.DataFrame(index=snapshot.index)
        # Size: log(mcap)
        if 'mcap' in snapshot.columns:
            controls['log_size'] = np.log(snapshot['mcap'].replace(0, np.nan).replace(-np.inf, np.nan))
        # Industry: if available
        if 'industry' in snapshot.columns:
            ind_dummies = pd.get_dummies(snapshot['industry'].astype(str), prefix='ind', drop_first=True)
            controls = pd.concat([controls, ind_dummies], axis=1)
        # If no controls available, skip
        if controls.empty or controls.shape[1] == 0:
            return snapshot[factor_col]
        # Neutralize
        factor = snapshot[factor_col].copy()
        config = {
            'winsorize': {'method': 'mad', 'n_mad': 5.0, 'level': None},
            'standardize': {'method': 'zscore', 'level': None},
            'neutralize': {'strength': strength, 'min_obs': 10, 'enabled': True},
            'restandardize': True,
        }
        return preprocess_pipeline(factor, controls=controls, config=config)

    def _neutralize_snapshot_multi(self, snapshot, factor_cols, strength=0.5):
        """Neutralize multiple factor columns."""
        result = snapshot.copy()
        for col in factor_cols:
            if col in result.columns:
                result[col] = self._neutralize_snapshot(result, col, strength)
        return result

    def run(
        self,
        universe_filter: Callable[[pd.DataFrame, pd.DatetimeIndex, int], List[str]] = None,
        ranking_factor: str = "mcap",
        ascending: bool = True,
        composite_factors: Optional[List[tuple]] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        take_profit_stocks: bool = False,
        take_profit_threshold: float = 0.50,
        ranking_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
        factor_weights: Optional[Dict[str, float]] = None,
        neutralize: bool = False,
        neutralize_strength: float = 0.5,
        pit_universe: bool = False,
        delist_manager: Optional["DelistManager"] = None,
    ) -> BacktestResult:
        """
        执行截面回测

        Args:
            universe_filter: (factor_snapshot, all_stocks, rebalance_idx) → List[str] 选股函数
            ranking_factor: 排名因子列名 (单因子模式, 当 ranking_fn/composite_factors 都未提供时使用)
            ascending: True=升序(选最小) | False=降序(选最大)
            composite_factors: [(factor_name, ascending), ...] 多因子z-score复合排名,
                              例如 [('mcap', True), ('pb', True), ('max_ret', False)]
                              提供此参数时 ranking_factor/ascending 被忽略
            stop_loss: 组合层面止损 (-0.20 表示跌破初始资本的 80% 清仓)
            trailing_stop: 移动止损 (0.25 表示从峰值回撤 25% 时清仓).
                          与 stop_loss 独立: 两个条件任一触发即退出.
                          优势: 策略先翻倍再跌时能保护已获利润.
            take_profit_stocks: 是否对个股启用止盈
            take_profit_threshold: 个股止盈阈值 (相对于买入价)
            ranking_fn: 复合排名函数 (factor_snapshot → pd.Series), 返回每只股票的综合评分
                        如果提供了 ranking_fn, 将忽略 composite_factors 和 ranking_factor
            factor_weights: {因子名: 权重} — 仅当 composite_factors 启用时生效
                            默认 None = 等权; 提供时用加权和替代等权和
            neutralize: 是否对 ranking_factor 做截面中性化 (size+industry 回归取残差)
            neutralize_strength: 中性化强度 (0=不中性化, 0.5=移除50%暴露, 1.0=完全中性化)
            pit_universe: 是否启用 PIT 无偏 universe 过滤 (opt-in, 默认 False = 完全保持原行为).
                          True 时在每个调仓日剔除"截至该日已退市"的标的, 缓解幸存者偏差.
                          注意: 仅能剔除 panel 中已含的已退市标的; 完整无偏还需数据层纳入退市标的.
            delist_manager: 可选注入的 DelistManager (用于测试/自定义退市缓存). None 且
                            pit_universe=True 时惰性构造默认实例 (读 data_cache/delisted_stocks.csv,
                            无本地缓存才联网). pit_universe=False 时该参数与退市模块均不被触碰.

        Returns:
            BacktestResult 对象
        """
        equity = self.initial_capital
        equity_curve = [equity]
        monthly_returns_log = []
        positions_log = {}
        position_buy_dates: Dict[str, pd.Timestamp] = {}  # T+1 追踪
        turnover_log = []
        current_holdings = {}  # {stock: (buy_price, weight, buy_date)}
        equity_dates = []
        total_days = 0
        _stop_triggered = False
        _stop_date = None
        _peak_equity = self.initial_capital  # trailing_stop 峰值追踪

        # 预提取因子面板以便快速查询
        factor_cols = list(self.factors.columns)

        # PIT 无偏 universe: 惰性构造退市管理器 (仅 opt-in 时触碰退市模块/网络)
        _delist_mgr = None
        if pit_universe:
            if delist_manager is not None:
                _delist_mgr = delist_manager
            else:
                from tools.backtest_mvp.data.delisted import DelistManager as _DM
                _delist_mgr = _DM()

        for i, rebal_date in enumerate(self.rebalance_dates):
            if i >= len(self.rebalance_dates) - 1:
                break

            rebal_date = pd.Timestamp(rebal_date)
            next_date = pd.Timestamp(self.rebalance_dates[i + 1])
            month_start_equity = equity  # 记录月初权益

            # --- 第 1 步: 获取因子横截面 ---
            try:
                snapshot = self._get_factor_snapshot(rebal_date)
            except:
                continue

            if len(snapshot) == 0:
                continue

            available_stocks = list(snapshot.index)

            # --- 第 2 步: 过滤 (用户自定义 + 涨跌停) ---
            if universe_filter is not None:
                selected = universe_filter(snapshot, self.dates, i)
            else:
                selected = available_stocks

            # PIT 无偏 universe 过滤 (opt-in): 剔除截至调仓日已退市的标的
            if _delist_mgr is not None:
                dead_set = set(_delist_mgr.get_delisted_before(rebal_date))
                if dead_set:
                    selected = [s for s in selected if s not in dead_set]


            # 涨跌停过滤: 排除涨停股 (买不到)
            if self.price_limit_stocks and 'is_limit_up' in snapshot.columns:
                limit_up_stocks = set(snapshot[snapshot['is_limit_up'] == True].index)
                if limit_up_stocks:
                    selected = [s for s in selected if s not in limit_up_stocks]

            # --- 第 2.5 步: 中性化 (可选) ---
            if neutralize and ranking_factor in snapshot.columns:
                snapshot_neut = snapshot.copy()
                snapshot_neut[ranking_factor] = self._neutralize_snapshot(
                    snapshot, ranking_factor, strength=neutralize_strength
                )
                snapshot = snapshot_neut

            # --- 第 3 步: 排名选股 (ranking_fn > composite_factors > ranking_factor) ---
            max_pick = self.n_stocks * 2 if self.price_limit_stocks else self.n_stocks
            picked = selected[:self.n_stocks]  # fallback

            if ranking_fn is not None:
                # 复合排名函数 (callable)
                scores = ranking_fn(snapshot.loc[selected])
                valid = scores.dropna()
                if len(valid) > 0:
                    picked = valid.nlargest(self.n_stocks).index.tolist()
            elif composite_factors is not None:
                # 多因子 z-score 复合排名
                factor_names = [f for f, _ in composite_factors]
                available = [f for f in factor_names if f in snapshot.columns]
                if len(available) > 0:
                    sub = snapshot.loc[selected][available].copy()
                    for col in available:
                        mean_val = sub[col].mean()
                        std_val = sub[col].std()
                        if std_val and std_val > 0:
                            sub[col] = (sub[col] - mean_val) / std_val
                        else:
                            sub[col] = 0
                    for f, asc in composite_factors:
                        if f in sub.columns and not asc:
                            sub[f] = -sub[f]
                    # 复合: 等权 sum 或 加权 sum
                    if factor_weights:
                        w_sum = pd.Series(0.0, index=sub.index)
                        total_w = 0.0
                        for f, asc in composite_factors:
                            if f in sub.columns:
                                w = factor_weights.get(f, 1.0 / len(composite_factors))
                                w_sum += sub[f] * w
                                total_w += w
                        if total_w > 0:
                            composite = w_sum / total_w  # normalize
                        else:
                            composite = sub.sum(axis=1, skipna=True)
                    else:
                        composite = sub.sum(axis=1, skipna=True)
                    ranked = composite.nsmallest(max_pick)
                    picked = ranked.index[:self.n_stocks].tolist()
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

            # --- 第 4 步: 等权分配 ---
            weight = 1.0 / len(picked)

            # --- 第 5 步: 计算持仓期收益率 ---
            rebal_ts = pd.Timestamp(rebal_date)
            next_ts = pd.Timestamp(next_date)
            period_dates = [d for d in self.dates
                           if d > rebal_ts and d <= next_ts]
            if not period_dates:
                continue

            period_returns = pd.DataFrame(0.0, index=period_dates, columns=picked)

            for j, date in enumerate(period_dates):
                ts = pd.Timestamp(date)
                # 获取当天收益率
                r = self._get_daily_return(ts, picked)

                # 如果所有返回都是 0/NaN (新上市或无交易), 跳过
                if r.abs().sum() == 0:
                    period_returns.loc[date] = 0
                    continue

                # 手续费 (调仓第一天)
                if j == 0:
                    equity *= (1 - self.commission)

                # 计算组合当天收益
                portfolio_return = (r.fillna(0) * weight).sum()
                period_returns.loc[date] = r.fillna(0).values

                # 滑点 (仅在调仓日: 反映买卖冲击成本)
                if j == 0:
                    new_picked = set(picked)
                    old_picked = set()
                    if i > 0:
                        prev_key = pd.Timestamp(self.rebalance_dates[i - 1])
                        if prev_key in positions_log:
                            old_picked = set(positions_log[prev_key].keys())
                    replaced_frac = len(new_picked - old_picked) / max(len(picked), 1)
                    portfolio_return -= self.slippage * replaced_frac

                # 更新权益
                equity *= (1 + portfolio_return)
                equity_curve.append(equity)
                equity_dates.append(ts)
                total_days += 1

                # --- 组合止损检查 (日频, 同日精度) ---
                # 更新峰值
                if trailing_stop and equity > _peak_equity:
                    _peak_equity = equity

                # 固定止损 (对初始资本)
                if stop_loss and equity / self.initial_capital <= (1 + stop_loss):
                    _stop_triggered = True
                    _stop_date = ts
                    break

                # 移动止损 (对峰值回撤)
                if trailing_stop and _peak_equity > 0 and \
                   equity / _peak_equity <= (1 - trailing_stop):
                    _stop_triggered = True
                    _stop_date = ts
                    break

            # 月度收益
            monthly_returns_log.append(equity / month_start_equity - 1)

            # --- 第 6 步: 记录持仓和换手 ---
            positions_log[rebal_ts] = {s: weight for s in picked}

            # 记录买入日期 (T+1): 新入场 → rebal_date, 保留 → 原 buy_date
            for s in picked:
                if s not in position_buy_dates:
                    position_buy_dates[s] = rebal_ts

            # 清理已卖出的持仓的 buy_date
            for s in list(position_buy_dates.keys()):
                if s not in picked:
                    del position_buy_dates[s]

            # 换手率 (与上月持仓对比)
            if i > 0:
                prev_key = pd.Timestamp(self.rebalance_dates[i-1])
                if prev_key in positions_log:
                    prev_picked = set(positions_log[prev_key].keys())
                    new_picked = set(picked)
                    turnover = len(new_picked - prev_picked) / max(len(picked), 1)
                    turnover_log.append(turnover)

            # --- 第 7 步: 检查组合止损 ---
            if _stop_triggered:
                break

        # --- 收尾 ---
        if not _stop_triggered:
            equity *= (1 - self.commission)

        # --- 构建权益曲线 (用 equity_dates 对 equity_curve) ---
        curve_data = list(zip(equity_dates, equity_curve[1:]))
        if len(curve_data) == 0:
            # 没有交易发生, 返回平线
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

        # 最大回撤
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()

        calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
        win_rate = (monthly_ret > 0).sum() / n_months if n_months > 0 else 0
        avg_turnover = np.mean(turnover_log) if turnover_log else 0

        # --- IC (Information Coefficient) calculation ---
        # For each rebalance date, compute Spearman rank correlation
        # between the ranking factor and the forward period return.
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

        # --- Quantile spread (Q5 - Q1) ---
        q_spread = self._compute_quantile_spread(ranking_factor, ranking_fn, composite_factors, factor_weights)

        # --- Monthly IC heatmap ---
        ic_heatmap = compute_monthly_ic_heatmap(ic_series)

        # --- Max drawdown recovery time ---
        recovery_days = _compute_max_drawdown_recovery_time(equity_series)

        # --- Rolling Sharpe ---
        rolling_sharpe_series = _compute_rolling_sharpe(monthly_ret, window=12)

        # --- Turnover attribution ---
        turnover_attr = _compute_turnover_attribution(
            turnover_log, monthly_ret
        )

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

        # --- robust_start_check: 熊市启动警告 ---
        if n_months < 3:
            print(f"  ⚠️ [robust_start] 回测仅运行 {n_months} 个月, "
                  f"不足以评估策略长期表现。可能是在熊市中提前止损。")
        elif len(monthly_ret) >= 2:
            # 检查前 3 个月是否持续下跌
            early_rets = monthly_ret[:min(3, len(monthly_ret))]
            if early_rets.mean() < -0.02:  # 月均亏损 > 2%
                print(f"  ⚠️ [robust_start] 前 {len(early_rets)} 个月月均收益 "
                      f"{early_rets.mean()*100:.1f}% (连续亏损), "
                      f"回测起点可能处于熊市阶段。结果可能低估策略长期表现。")

        return result

    def _compute_ic_series(
        self,
        ranking_factor: str,
        ranking_fn,
        composite_factors,
        factor_weights,
    ) -> pd.Series:
        """Compute Spearman rank IC for each rebalance period.

        For each rebalance date i, correlate the signal at date i with
        the realized return over [date_i, date_{i+1}).
        Returns a Series indexed by rebalance date.
        """
        # IC uses pandas rank correlation, no scipy needed
        ic_values = []
        ic_dates = []
        for i, rebal_date in enumerate(self.rebalance_dates):
            if i >= len(self.rebalance_dates) - 1:
                break
            next_date = self.rebalance_dates[i + 1]
            rebal_ts = pd.Timestamp(rebal_date)
            next_ts = pd.Timestamp(next_date)

            try:
                snapshot = self._get_factor_snapshot(rebal_ts)
            except Exception:
                continue
            if len(snapshot) == 0:
                continue

            # Compute signal scores for all stocks in snapshot
            if ranking_fn is not None:
                scores = ranking_fn(snapshot)
            elif composite_factors is not None:
                available = [f for f, _ in composite_factors if f in snapshot.columns]
                if not available:
                    continue
                sub = snapshot[available].copy()
                for col in available:
                    mu, sigma = sub[col].mean(), sub[col].std()
                    sub[col] = (sub[col] - mu) / sigma if sigma and sigma > 0 else 0
                for f, asc in composite_factors:
                    if f in sub.columns and not asc:
                        sub[f] = -sub[f]
                if factor_weights:
                    w_sum = pd.Series(0.0, index=sub.index)
                    total_w = 0.0
                    for f, _ in composite_factors:
                        if f in sub.columns:
                            w = factor_weights.get(f, 1.0 / len(composite_factors))
                            w_sum += sub[f] * w
                            total_w += w
                    scores = w_sum / total_w if total_w > 0 else sub.sum(axis=1, skipna=True)
                else:
                    scores = sub.sum(axis=1, skipna=True)
            elif ranking_factor in snapshot.columns:
                scores = snapshot[ranking_factor]
            else:
                continue

            # Get forward returns
            period_dates = [d for d in self.dates if d > rebal_ts and d <= next_ts]
            if not period_dates:
                continue
            fwd_returns = self._get_period_returns(rebal_ts, next_ts, list(snapshot.index))
            if fwd_returns.empty:
                continue
            period_ret = fwd_returns.sum(axis=0)  # cumulative return over period

            # Align scores and returns
            common = scores.dropna().index.intersection(period_ret.dropna().index)
            if len(common) < 5:
                continue
            s_scores = scores.loc[common].rank()
            s_returns = period_ret.loc[common].rank()
            if s_scores.std() == 0 or s_returns.std() == 0:
                continue
            corr = float(s_scores.corr(s_returns))
            if pd.notna(corr):
                ic_values.append(corr)
                ic_dates.append(rebal_ts)

        if ic_dates:
            return pd.Series(ic_values, index=pd.Index(ic_dates, name="date"))
        return pd.Series(dtype=float)

    def compute_ic_decay(
        self,
        ranking_factor: str = "mcap",
        lags: tuple = (1, 5, 10, 20),
    ) -> dict[int, float]:
        """Compute IC at multiple forward-return lags.

        For each lag L, correlate the signal at rebalance date i with
        the return over the next L trading days.

        Returns {lag: mean_ic} mapping.
        """
        result = {}
        for lag in lags:
            ic_values = []
            for i, rebal_date in enumerate(self.rebalance_dates):
                rebal_ts = pd.Timestamp(rebal_date)
                # Get L trading days after rebal_date
                future_dates = [d for d in self.dates if d > rebal_ts][:lag]
                if len(future_dates) < lag:
                    continue
                try:
                    snapshot = self._get_factor_snapshot(rebal_ts)
                except Exception:
                    continue
                if len(snapshot) == 0 or ranking_factor not in snapshot.columns:
                    continue
                scores = snapshot[ranking_factor].dropna()
                # Get cumulative return over lag days
                fwd_returns = self._get_period_returns(
                    rebal_ts, future_dates[-1], list(scores.index)
                )
                if fwd_returns.empty:
                    continue
                period_ret = fwd_returns.sum(axis=0)
                common = scores.index.intersection(period_ret.dropna().index)
                if len(common) < 5:
                    continue
                corr = float(scores.loc[common].rank().corr(period_ret.loc[common].rank()))
                if pd.notna(corr):
                    ic_values.append(corr)
            result[lag] = float(np.mean(ic_values)) if ic_values else 0.0
        return result

    def _compute_quantile_spread(
        self,
        ranking_factor: str,
        ranking_fn,
        composite_factors,
        factor_weights,
        n_quantiles: int = 5,
    ) -> float:
        """Compute average Q5-Q1 return spread across rebalance dates.

        For each rebalance period, sort stocks into N quantiles by signal,
        compute each quantile's return, and return mean(Q5 - Q1).
        """
        spreads = []
        for i, rebal_date in enumerate(self.rebalance_dates):
            if i >= len(self.rebalance_dates) - 1:
                break
            next_date = self.rebalance_dates[i + 1]
            rebal_ts = pd.Timestamp(rebal_date)
            next_ts = pd.Timestamp(next_date)

            try:
                snapshot = self._get_factor_snapshot(rebal_ts)
            except Exception:
                continue
            if len(snapshot) == 0:
                continue

            # Compute signal scores
            if ranking_fn is not None:
                scores = ranking_fn(snapshot)
            elif composite_factors is not None:
                available = [f for f, _ in composite_factors if f in snapshot.columns]
                if not available:
                    continue
                sub = snapshot[available].copy()
                for col in available:
                    mu, sigma = sub[col].mean(), sub[col].std()
                    sub[col] = (sub[col] - mu) / sigma if sigma and sigma > 0 else 0
                for f, asc in composite_factors:
                    if f in sub.columns and not asc:
                        sub[f] = -sub[f]
                scores = sub.sum(axis=1, skipna=True)
            elif ranking_factor in snapshot.columns:
                scores = snapshot[ranking_factor]
            else:
                continue

            valid_scores = scores.dropna()
            if len(valid_scores) < n_quantiles * 2:
                continue

            # Get forward returns
            fwd_returns = self._get_period_returns(rebal_ts, next_ts, list(valid_scores.index))
            if fwd_returns.empty:
                continue
            period_ret = fwd_returns.sum(axis=0)

            # Align
            common = valid_scores.index.intersection(period_ret.dropna().index)
            if len(common) < n_quantiles * 2:
                continue
            s = valid_scores.loc[common]
            r = period_ret.loc[common]

            # Sort into quantiles
            try:
                q_labels = pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue
            q_returns = r.groupby(q_labels).mean()
            if len(q_returns) >= 2:
                spreads.append(float(q_returns.iloc[-1] - q_returns.iloc[0]))

        return float(np.mean(spreads)) if spreads else 0.0
