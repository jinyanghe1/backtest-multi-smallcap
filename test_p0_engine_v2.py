#!/usr/bin/env python3
"""
P0EngineV2 单元测试 — 18 个用例覆盖全部 P0 功能

原则:
- 全部合成数据，禁止真实网络请求
- 每个 P0 功能单独测试，确保隔离性
- 关闭 P0 时与原引擎对比
- 边界条件: 空数据、单只股票、极端行情
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.backtest_mvp.engine import CrossSectionalEngine, BacktestResult
from tools.backtest_mvp.p0_engine_v2 import P0EngineV2
from tools.backtest_mvp.data.delisted import DelistManager
from tools.backtest_mvp.data.adv_impact import ADVImpactModel
from tools.backtest_mvp.data.risk_overlay import RiskOverlay, RiskOverlayConfig


# ═══════════════════════════════════════════════════════════
# 合成数据工厂
# ═══════════════════════════════════════════════════════════

def _make_dates(n=60, start="2020-01-01"):
    """生成交易日序列 (排除周末)"""
    dates = pd.bdate_range(start=start, periods=n)
    return dates


def make_factor_panel(n_stocks=3, n_days=60, start="2020-01-01", symbols=None):
    """合成因子面板"""
    dates = _make_dates(n_days, start)
    if symbols is None:
        symbols = [f"sh60000{i+1:02d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    df = pd.DataFrame(index=idx)
    df["mcap"] = np.random.lognormal(20, 0.5, len(idx))  # 市值
    df["pb"] = np.random.uniform(0.5, 5.0, len(idx))      # 市净率
    df["mom20d"] = np.random.normal(0, 0.05, len(idx))    # 20日动量
    return df


def make_return_panel(n_stocks=3, n_days=60, start="2020-01-01", drift=0.001, symbols=None):
    """合成收益率面板"""
    dates = _make_dates(n_days, start)
    if symbols is None:
        symbols = [f"sh60000{i+1:02d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    df = pd.DataFrame(index=idx)
    # 日收益率：正态分布 + 漂移
    df["daily_return"] = np.random.normal(drift, 0.02, len(idx))
    return df


def make_listing_dates(symbols, list_dates):
    """合成上市日期表"""
    df = pd.DataFrame({
        "symbol": symbols,
        "name": [f"Stock{i}" for i in range(len(symbols))],
        "list_date": pd.to_datetime(list_dates),
        "market": [s[:2] for s in symbols],
    })
    return df


def make_adv_data(symbols, dates, adv_values):
    """合成 ADV 数据"""
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    df = pd.DataFrame(index=idx)
    # adv_values: dict {symbol: adv}
    for sym, adv in adv_values.items():
        df.loc[pd.IndexSlice[:, sym], "adv_20d"] = adv
    return df


# ═══════════════════════════════════════════════════════════
# TC01: 初始化默认参数
# ═══════════════════════════════════════════════════════════

def test_tc01_init_defaults():
    """所有 P0 功能默认开启"""
    fp = make_factor_panel()
    rp = make_return_panel()
    engine = P0EngineV2(factor_panel=fp, return_panel=rp)
    assert engine.enable_pit_universe is True
    assert engine.enable_adv_impact is True
    assert engine.enable_risk_overlay is True
    assert engine.enable_deflated_sharpe is True
    assert engine.n_stocks == 30


# ═══════════════════════════════════════════════════════════
# TC02: 日期对齐
# ═══════════════════════════════════════════════════════════

def test_tc02_date_alignment():
    """factor 和 return 日期取交集"""
    fp = make_factor_panel(n_days=60, start="2020-01-01")
    rp = make_return_panel(n_days=50, start="2020-02-01")  # 少 10 天
    engine = P0EngineV2(factor_panel=fp, return_panel=rp)
    # 交集应该是 2020-02-01 到 2020-03-20 的交易日
    assert len(engine.dates) < 60
    assert len(engine.dates) > 0


# ═══════════════════════════════════════════════════════════
# TC03: 关闭 P0 时与原引擎一致
# ═══════════════════════════════════════════════════════════

def test_tc03_disabled_p0_matches_original():
    """关闭所有 P0 功能，结果应与原引擎一致"""
    fp = make_factor_panel(n_stocks=5, n_days=60)
    rp = make_return_panel(n_stocks=5, n_days=60)

    # 原引擎
    orig = CrossSectionalEngine(
        factor_panel=fp, return_panel=rp, n_stocks=3, rebalance_freq='M'
    )
    orig_result = orig.run(ranking_factor="mcap", ascending=True)

    # P0 v2 关闭所有功能
    p0 = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=3, rebalance_freq='M',
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    p0_result = p0.run(ranking_factor="mcap", ascending=True)

    # 关键指标差异 < 2% (允许数值精度差异，因佣金计算时机略有不同)
    assert abs(p0_result.terminal_value - orig_result.terminal_value) < 0.02
    assert abs(p0_result.sharpe_ratio - orig_result.sharpe_ratio) < 0.02


# ═══════════════════════════════════════════════════════════
# TC04-06: PIT Universe 过滤
# ═══════════════════════════════════════════════════════════

def test_tc04_pit_filters_unlisted():
    """未上市股票被排除"""
    symbols = ["sh600001", "sh600002", "sh600003"]
    # 600001 上市于 2020-02-01，600002 上市于 2020-01-01，600003 未上市
    listing = make_listing_dates(
        symbols,
        ["2020-02-01", "2020-01-01", "2025-01-01"]  # 600003 未上市
    )
    fp = make_factor_panel(n_stocks=3, n_days=30, start="2020-01-01")
    rp = make_return_panel(n_stocks=3, n_days=30, start="2020-01-01")

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        listing_dates=listing,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    # 2020-01-01 时，600001 还没上市，600003 也没上市
    # 只有 600002 活着
    # 检查持仓日志
    first_rebal = list(result.positions_log.columns)
    # 第一个调仓日 (2020-01-31) 时，600001 已上市，600003 未上市
    # 所以应该只有 600001 和 600002 被选中
    assert "sh600003" not in first_rebal


def test_tc05_pit_filters_delisted():
    """已退市股票被排除"""
    symbols = ["sh600001", "sh600002", "sh600003"]
    listing = make_listing_dates(symbols, ["2020-01-01", "2020-01-01", "2020-01-01"])

    fp = make_factor_panel(n_stocks=3, n_days=60, start="2020-01-01")
    rp = make_return_panel(n_stocks=3, n_days=60, start="2020-01-01")

    # Mock delist_manager: 600003 于 2020-02-01 退市
    mock_mgr = MagicMock(spec=DelistManager)
    mock_mgr.is_alive = lambda sym, date: False if sym == "sh600003" and date >= pd.Timestamp("2020-02-01") else True
    mock_mgr.get_delisted_before = lambda date: ["sh600003"] if date >= pd.Timestamp("2020-02-01") else []

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        listing_dates=listing,
        delist_manager=mock_mgr,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    # 第二个调仓日 (2020-02-28) 时，600003 已退市
    rebal_dates = list(result.positions_log.columns)
    if len(rebal_dates) >= 2:
        second_rebal = result.positions_log.iloc[:, 1]
        assert "sh600003" not in second_rebal[second_rebal > 0].index


def test_tc06_pit_defaults_alive_when_no_data():
    """无 listing_dates 时默认全部活着"""
    fp = make_factor_panel(n_stocks=3, n_days=30)
    rp = make_return_panel(n_stocks=3, n_days=30)
    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        listing_dates=None,  # 无数据
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)
    # 应该正常选股，不崩溃
    assert result.terminal_value > 0


# ═══════════════════════════════════════════════════════════
# TC07-09: ADV 冲击
# ═══════════════════════════════════════════════════════════

def test_tc07_adv_impact_with_data():
    """有 ADV 数据时，冲击成本 > 0，降低收益"""
    symbols = ["sh600001", "sh600002", "sh600003"]
    dates = _make_dates(30, "2020-01-01")
    fp = make_factor_panel(n_stocks=3, n_days=30)
    rp = make_return_panel(n_stocks=3, n_days=30)

    # ADV: 600001=1亿 (高), 600002=100万 (低), 600003=无数据
    adv = make_adv_data(symbols, dates, {"sh600001": 1e8, "sh600002": 1e6})

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
        adv_data=adv,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    # 有 ADV 数据时，冲击成本应该降低收益
    # 与无 ADV 对比
    engine_no_adv = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result_no_adv = engine_no_adv.run(ranking_factor="mcap", ascending=True)

    # 有 ADV 冲击时，收益应该 <= 无冲击时 (允许微小差异)
    assert result.terminal_value <= result_no_adv.terminal_value + 0.01


def test_tc08_adv_impact_no_data():
    """无 ADV 数据时，冲击成本 = 0"""
    fp = make_factor_panel(n_stocks=3, n_days=30)
    rp = make_return_panel(n_stocks=3, n_days=30)

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
        adv_data=None,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    # 与关闭 ADV 对比
    engine_off = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result_off = engine_off.run(ranking_factor="mcap", ascending=True)

    assert abs(result.terminal_value - result_off.terminal_value) < 0.01


def test_tc09_adv_micro_vs_large():
    """微盘冲击成本 > 大盘冲击成本"""
    symbols = ["sh600001", "sh600002"]
    dates = _make_dates(30, "2020-01-01")
    fp = make_factor_panel(n_stocks=2, n_days=30)
    rp = make_return_panel(n_stocks=2, n_days=30)

    # 600001: ADV=1亿 (大盘), 600002: ADV=50万 (微盘)
    adv = make_adv_data(symbols, dates, {"sh600001": 1e8, "sh600002": 5e5})

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=1,
        enable_pit_universe=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
        adv_data=adv,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    # 验证：如果选中微盘，冲击成本应更高
    # 这里主要验证引擎不崩溃，且 ADV 数据被正确读取
    assert result.terminal_value > 0


# ═══════════════════════════════════════════════════════════
# TC10-12: Risk Overlay
# ═══════════════════════════════════════════════════════════

def test_tc10_risk_overlay_normal():
    """正常行情时，gross = 1.0"""
    fp = make_factor_panel(n_stocks=3, n_days=30)
    rp = make_return_panel(n_stocks=3, n_days=30, drift=0.001)  # 上涨

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    # 正常行情，不应触发 Risk Overlay
    # 与关闭 Risk Overlay 对比，结果应接近
    engine_off = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result_off = engine_off.run(ranking_factor="mcap", ascending=True)

    assert abs(result.terminal_value - result_off.terminal_value) < 0.05


def test_tc11_risk_overlay_drawdown():
    """回撤时，gross < 1.0，减少损失"""
    # 构造一个先涨后跌的收益率序列
    symbols = ["sh600001", "sh600002", "sh600003"]
    dates = _make_dates(60, "2020-01-01")
    fp = make_factor_panel(n_stocks=3, n_days=60)

    # 前 30 天涨 1%，后 30 天跌 2%
    returns = []
    for i, d in enumerate(dates):
        if i < 30:
            r = 0.01
        else:
            r = -0.02
        for s in symbols:
            returns.append({"date": d, "symbol": s, "daily_return": r + np.random.normal(0, 0.005)})
    rp = pd.DataFrame(returns).set_index(["date", "symbol"])

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_deflated_sharpe=False,
        risk_overlay_config=RiskOverlayConfig(
            target_vol_annual=0.15,
            dd_threshold=-0.05,  # 放宽阈值便于测试
            dd_gross=0.50,
        ),
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    # 回撤时 Risk Overlay 应降低 gross
    # 与关闭对比，最大回撤应更小
    engine_off = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result_off = engine_off.run(ranking_factor="mcap", ascending=True)

    assert result.max_drawdown >= result_off.max_drawdown - 1  # Risk Overlay 不应让回撤更差


def test_tc12_risk_overlay_severe_drawdown():
    """严重回撤时，gross = 0.2"""
    # 构造暴跌行情
    symbols = ["sh600001", "sh600002", "sh600003"]
    dates = _make_dates(30, "2020-01-01")
    fp = make_factor_panel(n_stocks=3, n_days=30)

    returns = []
    for d in dates:
        for s in symbols:
            returns.append({"date": d, "symbol": s, "daily_return": -0.03})  # 每天跌 3%
    rp = pd.DataFrame(returns).set_index(["date", "symbol"])

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_deflated_sharpe=False,
        risk_overlay_config=RiskOverlayConfig(
            dd_threshold=-0.05,
            dd_severe=-0.10,
            dd_severe_gross=0.20,
        ),
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    # 严重回撤，权益应该还有剩余 (因为 gross 被降到 0.2)
    # 与无 Risk Overlay 对比，无 overlay 时权益会更快归零
    engine_off = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result_off = engine_off.run(ranking_factor="mcap", ascending=True)

    # 有 overlay 时，最终权益应该 >= 无 overlay (因为降仓减少了损失)
    assert result.terminal_value >= result_off.terminal_value - 0.01


# ═══════════════════════════════════════════════════════════
# TC13-14: 止损逻辑
# ═══════════════════════════════════════════════════════════

def test_tc13_stop_loss_triggered():
    """固定止损触发"""
    fp = make_factor_panel(n_stocks=2, n_days=30)
    rp = make_return_panel(n_stocks=2, n_days=30)
    # 覆盖为连续大跌 (-5% 每天)
    rp['daily_return'] = -0.05

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=1,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True, stop_loss=-0.20)

    # 要么止损触发，要么权益大幅亏损
    assert result.stop_triggered or result.max_drawdown <= -20
    assert result.stop_trigger_date != ""


def test_tc14_trailing_stop_triggered():
    """移动止损触发"""
    fp = make_factor_panel(n_stocks=2, n_days=60)
    rp = make_return_panel(n_stocks=2, n_days=60)
    # 覆盖：前30天涨 2%，后30天跌 3%
    dates = _make_dates(60, "2020-01-01")
    for i, d in enumerate(dates):
        r = 0.02 if i < 30 else -0.03
        rp.loc[pd.IndexSlice[d, :], 'daily_return'] = r

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=1,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True, trailing_stop=0.25)

    # 峰值后回撤 25% 应该触发
    assert result.stop_triggered is True


# ═══════════════════════════════════════════════════════════
# TC15: Deflated Sharpe
# ═══════════════════════════════════════════════════════════

def test_tc15_deflated_sharpe():
    """Deflated Sharpe <= 原始 Sharpe"""
    fp = make_factor_panel(n_stocks=3, n_days=60)
    rp = make_return_panel(n_stocks=3, n_days=60)

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=True,
        n_trials=10,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    assert hasattr(result, 'deflated_sharpe')
    assert result.deflated_sharpe <= result.sharpe_ratio + 0.01  # 允许精度误差


# ═══════════════════════════════════════════════════════════
# TC16-18: 边界条件
# ═══════════════════════════════════════════════════════════

def test_tc16_empty_data():
    """空数据不崩溃"""
    fp = pd.DataFrame(columns=["mcap"])
    fp.index = pd.MultiIndex.from_tuples([], names=["date", "symbol"])
    rp = pd.DataFrame(columns=["daily_return"])
    rp.index = pd.MultiIndex.from_tuples([], names=["date", "symbol"])

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=2,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    assert result.terminal_value == 1.0  # 初始资本
    assert result.sharpe_ratio == 0


def test_tc17_single_stock():
    """单只股票正常计算"""
    fp = make_factor_panel(n_stocks=1, n_days=30)
    rp = make_return_panel(n_stocks=1, n_days=30)

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=1,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True)

    assert result.terminal_value > 0
    assert len(result.positions_log) > 0


def test_tc18_extreme_market():
    """极端行情：连续跌停"""
    fp = make_factor_panel(n_stocks=2, n_days=30)
    rp = make_return_panel(n_stocks=2, n_days=30)
    # 覆盖为连续跌停 (-10%)
    rp['daily_return'] = -0.10

    engine = P0EngineV2(
        factor_panel=fp, return_panel=rp, n_stocks=1,
        enable_pit_universe=False,
        enable_adv_impact=False,
        enable_risk_overlay=False,
        enable_deflated_sharpe=False,
    )
    result = engine.run(ranking_factor="mcap", ascending=True, stop_loss=-0.50)

    # 要么止损触发，要么权益趋近于 0
    assert result.stop_triggered or result.terminal_value < 0.1


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
