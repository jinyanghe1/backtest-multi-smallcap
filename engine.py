"""
截面回测引擎 — Cross-Sectional Walk-Forward Backtest Engine
=============================================================
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

    def run(
        self,
        universe_filter: Callable[[pd.DataFrame, pd.DatetimeIndex, int], List[str]] = None,
        ranking_factor: str = "mcap",
        ascending: bool = True,
        stop_loss: Optional[float] = None,
        take_profit_stocks: bool = False,
        take_profit_threshold: float = 0.50,
        ranking_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    ) -> BacktestResult:
        """
        执行截面回测

        Args:
            universe_filter: (factor_snapshot, all_stocks, rebalance_idx) → List[str] 选股函数
            ranking_factor: 排名因子列名 (当 ranking_fn 为 None 时使用)
            ascending: True=升序(选最小) | False=降序(选最大)
            stop_loss: 组合层面止损 (-0.20 表示跌破初始的 80% 清仓)
            take_profit_stocks: 是否对个股启用止盈
            take_profit_threshold: 个股止盈阈值 (相对于买入价)
            ranking_fn: 复合排名函数 (factor_snapshot → pd.Series), 返回每只股票的综合评分
                        如果提供了 ranking_fn, 将忽略 ranking_factor 和 ascending

        Returns:
            BacktestResult 对象
        """
        equity = self.initial_capital
        equity_curve = [equity]
        monthly_returns_log = []
        positions_log = {}
        turnover_log = []
        current_holdings = {}  # {stock: (buy_price, weight, buy_date)}
        equity_dates = []
        total_days = 0

        # 预提取因子面板以便快速查询
        factor_cols = list(self.factors.columns)

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

            # --- 第 2 步: 过滤 (用户自定义) ---
            if universe_filter is not None:
                selected = universe_filter(snapshot, self.dates, i)
            else:
                # 默认: 所有股票
                selected = available_stocks

            # --- 第 3 步: 排名选股 ---
            if ranking_fn is not None:
                # 复合排名: ranking_fn(snapshot) → pd.Series of scores
                scores = ranking_fn(snapshot.loc[selected])
                valid = scores.dropna()
                if len(valid) > 0:
                    # 选得分最高的 n_stocks 只 (降序)
                    picked = valid.nlargest(self.n_stocks).index.tolist()
                else:
                    picked = selected[:self.n_stocks]
            elif ranking_factor in snapshot.columns:
                valid = snapshot.loc[selected][ranking_factor].dropna()
                if ascending:
                    picked = valid.nsmallest(self.n_stocks).index.tolist()
                else:
                    picked = valid.nlargest(self.n_stocks).index.tolist()
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

                # 滑点
                portfolio_return -= self.slippage * len(picked) / (
                    len(picked) * 252 / len(self.rebalance_dates))

                # 更新权益
                equity *= (1 + portfolio_return)
                equity_curve.append(equity)
                equity_dates.append(ts)
                total_days += 1

            # 月度收益
            monthly_returns_log.append(equity / month_start_equity - 1)

            # --- 第 6 步: 记录持仓和换手 ---
            positions_log[rebal_ts] = {s: weight for s in picked}

            # 换手率 (与上月持仓对比)
            if i > 0:
                prev_key = pd.Timestamp(self.rebalance_dates[i-1])
                if prev_key in positions_log:
                    prev_picked = set(positions_log[prev_key].keys())
                    new_picked = set(picked)
                    turnover = len(new_picked - prev_picked) / max(len(picked), 1)
                    turnover_log.append(turnover)

            # --- 第 7 步: 检查组合止损 ---
            if stop_loss and equity / self.initial_capital <= (1 + stop_loss):
                # 清仓, 等权现金
                break

        # --- 收尾 ---
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
        )
        return result
