"""
PIT 无偏 universe 引擎开关测试 (Tier 4 / roadmap UA1).

验证 engine.run() 新增的 opt-in 参数 pit_universe / delist_manager:
  1. 默认关闭 → 与不传参完全一致 (零回归).
  2. 开启 + 注入 DelistManager → 在每个调仓日剔除"截至该日已退市"的标的.
  3. 注入的 manager 离线可用 (不联网), 空黑名单为 no-op, 与用户 universe_filter 组合正确.

全部使用合成数据 + 临时退市 CSV, 离线确定性, 不触碰网络。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.backtest_mvp.engine import CrossSectionalEngine
from tools.backtest_mvp.data.delisted import DelistManager


SYMBOLS = ["sz000001", "sz000002", "sz000003", "sz000004"]
DELISTED = "sz000003"


def _build_panels(n_days: int = 6):
    """构造合成 factor_panel + return_panel (MultiIndex[date, symbol])。"""
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, SYMBOLS], names=["date", "symbol"])

    rng = np.random.RandomState(42)
    # mcap: 每只股票一个稳定量级, 保证 4 只都能进 (n_stocks>=4)
    base_mcap = {s: 10.0 + i for i, s in enumerate(SYMBOLS)}
    factor_panel = pd.DataFrame(
        {
            "mcap": [base_mcap[s] for (_, s) in idx],
            "pb": rng.uniform(1.0, 3.0, size=len(idx)),
        },
        index=idx,
    )
    # 恒定正收益, 保证 r.abs().sum() != 0 (否则引擎跳过该调仓不记录持仓)
    return_panel = pd.DataFrame({"daily_return": 0.001}, index=idx)
    return dates, factor_panel, return_panel


def _make_delist_manager(mapping: dict) -> DelistManager:
    """用临时 CSV 构造离线 DelistManager。mapping: {symbol: delist_date|None}。"""
    tmp = Path(tempfile.mkdtemp()) / "delisted_stocks.csv"
    rows = []
    for sym, dd in mapping.items():
        rows.append(
            {
                "symbol": sym,
                "name": sym.upper(),
                "list_date": "2010-01-01",
                "delist_date": "" if dd is None else pd.Timestamp(dd).strftime("%Y-%m-%d"),
                "market": sym[:2],
            }
        )
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8-sig")
    return DelistManager(cache_path=tmp)


def _engine(n_days: int = 6):
    dates, fp, rp = _build_panels(n_days)
    eng = CrossSectionalEngine(
        factor_panel=fp,
        return_panel=rp,
        n_stocks=4,
        rebalance_freq="D",
        price_limit_stocks=False,
    )
    return dates, eng


def _held_symbols(result) -> set:
    """positions_log: index=symbol, columns=rebalance_date, value=weight。"""
    pl = result.positions_log
    if pl is None or pl.empty:
        return set()
    return set(pl.index[(pl != 0).any(axis=1)])


# ─────────────────────────── 向后兼容 ───────────────────────────

def test_default_off_equals_no_arg():
    """默认 pit_universe=False 与完全不传参数产生 byte-identical equity_curve。"""
    _, eng1 = _engine()
    _, eng2 = _engine()
    r_default = eng1.run()
    r_explicit = eng2.run(pit_universe=False)
    pd.testing.assert_series_equal(r_default.equity_curve, r_explicit.equity_curve)


def test_off_does_not_touch_delist_module():
    """pit_universe=False 时即使传了会抛错的哨兵 manager 也不被调用。"""

    class Boom:
        def get_delisted_before(self, *_a, **_k):
            raise AssertionError("delist manager must NOT be touched when pit_universe=False")

    _, eng = _engine()
    # 不应抛异常 (说明 Boom.get_delisted_before 从未被调用)
    result = eng.run(pit_universe=False, delist_manager=Boom())
    assert result.equity_curve is not None


# ─────────────────────────── PIT 生效 ───────────────────────────

def test_baseline_includes_delisted_name():
    """基线 (PIT off): 已退市标的仍被持有 (幸存者偏差存在)。"""
    _, eng = _engine()
    result = eng.run(pit_universe=False)
    assert DELISTED in _held_symbols(result)


def test_pit_excludes_delisted_on_and_after_delist_date():
    """PIT on: 退市标的在退市日 (含) 之后不再被持有, 之前仍持有。"""
    dates, eng = _engine()
    mgr = _make_delist_manager({DELISTED: dates[2]})
    result = eng.run(pit_universe=True, delist_manager=mgr)

    pl = result.positions_log
    assert DELISTED in pl.index, "退市前应至少被持有一次"
    # 退市前 (dates[0], dates[1]): 权重 > 0
    for d in dates[:2]:
        if d in pl.columns:
            assert pl.loc[DELISTED, d] > 0, f"{d} 退市前应被持有"
    # 退市日及之后 (dates[2:]): 权重 == 0
    for d in dates[2:]:
        if d in pl.columns:
            assert pl.loc[DELISTED, d] == 0, f"{d} 退市后不应被持有"


def test_pit_keeps_alive_names():
    """PIT on: 未退市标的不受影响, 全程被持有。"""
    dates, eng = _engine()
    mgr = _make_delist_manager({DELISTED: dates[2]})
    result = eng.run(pit_universe=True, delist_manager=mgr)
    held = _held_symbols(result)
    for s in SYMBOLS:
        if s != DELISTED:
            assert s in held, f"{s} 未退市, 应被持有"


def test_empty_blacklist_is_noop():
    """PIT on 但黑名单对当前区间为空 → 与 PIT off 完全一致。"""
    dates, eng1 = _engine()
    _, eng2 = _engine()
    # 退市日远在回测区间之后 → get_delisted_before 恒为空
    mgr = _make_delist_manager({DELISTED: "2099-01-01"})
    r_on = eng1.run(pit_universe=True, delist_manager=mgr)
    r_off = eng2.run(pit_universe=False)
    pd.testing.assert_series_equal(r_on.equity_curve, r_off.equity_curve)
    assert DELISTED in _held_symbols(r_on)


def test_pit_composes_with_user_universe_filter():
    """PIT 过滤与用户自定义 universe_filter 叠加: 用户先选子集, PIT 再剔除已退市。"""
    dates, eng = _engine()
    mgr = _make_delist_manager({DELISTED: dates[2]})

    # 用户 filter: 只保留 sz000003 + sz000004
    def uf(snapshot, all_dates, i):
        return [s for s in ["sz000003", "sz000004"] if s in snapshot.index]

    result = eng.run(pit_universe=True, delist_manager=mgr, universe_filter=uf)
    pl = result.positions_log
    # sz000004 始终在; sz000003 退市后被 PIT 剔除
    assert "sz000004" in _held_symbols(result)
    for d in dates[2:]:
        if d in pl.columns and DELISTED in pl.index:
            assert pl.loc[DELISTED, d] == 0


# ─────────────────────── DelistManager 单元语义 ───────────────────────

def test_get_delisted_before_semantics():
    """get_delisted_before(date): 返回 delist_date <= date 的标的。"""
    mgr = _make_delist_manager({DELISTED: "2023-01-04", "sz000002": None})
    before = mgr.get_delisted_before("2023-01-03")
    on = mgr.get_delisted_before("2023-01-04")
    after = mgr.get_delisted_before("2023-01-10")
    assert DELISTED not in before          # 退市日之前 → 未死
    assert DELISTED in on                  # 退市日当天 (<=) → 已死
    assert DELISTED in after               # 之后 → 已死
    assert "sz000002" not in after         # delist_date=NaN → 永不退市


def test_injected_manager_is_offline():
    """注入的 manager 从本地 CSV 读取, 全程离线 (df 已加载, 不触发 fetch_all)。"""
    mgr = _make_delist_manager({DELISTED: "2023-01-04"})
    assert mgr.df is not None and len(mgr.df) >= 1
    # 直接查询不联网即可返回
    assert DELISTED in mgr.get_delisted_before("2023-06-01")


def test_multi_delist_dates_independent():
    """多标的不同退市日: 各自在自己的退市日后被剔除, 互不影响。"""
    dates, eng = _engine()
    mgr = _make_delist_manager({"sz000002": dates[1], DELISTED: dates[3]})
    result = eng.run(pit_universe=True, delist_manager=mgr)
    pl = result.positions_log
    # sz000002 在 dates[1] 后消失
    for d in dates[1:]:
        if d in pl.columns and "sz000002" in pl.index:
            assert pl.loc["sz000002", d] == 0
    # sz000003 在 dates[3] 前仍在, dates[3] 后消失
    if DELISTED in pl.index and dates[2] in pl.columns:
        assert pl.loc[DELISTED, dates[2]] > 0
    for d in dates[3:]:
        if d in pl.columns and DELISTED in pl.index:
            assert pl.loc[DELISTED, d] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
