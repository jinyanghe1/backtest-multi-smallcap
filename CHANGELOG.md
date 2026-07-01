# Changelog

本文件记录 backtest_mvp 的高层变更。因子级细节见 `FACTOR_LIBRARY.json`，
alpha 路线图进度见 `roadmap_alpha_uplift.json` 的 `progress_log`。

## 2026-07-01 — Phase 2：因子组合化 + 数据真实化

运行时：`~/.workbuddy/binaries/python/envs/default/bin/python`（Py3.13，
numpy2.5/pandas3.0，scipy/akshare 可用）。验证：131 tests pass
（`test_factors_p0..p4 + test_operators + test_shareholders`），零回归。

### 新增 (Added)
- **去相关 IC 加权因子组合工具**（roadmap UD3/UC3，commit `f15002e`）
  - `factors/combiner.py`：`combine_factors()` —— 复用 Spearman IC → 按 |IC-IR|
    排序贪心去相关（`|corr|<max_corr`）→ 按 IC 符号对齐 →
    `equal`/`ic`/`ic_ir` 加权 → 截面 z-score 合成；
    `make_composite_strategy_def()` 产出可回测 `strategy_def`（`ranking_fn`），
    **严格无前视**（IC 权重仅用 `<= t-fwd_period` 的数据估计）。
  - `factor_report.py`：跑全 `FACTOR_REGISTRY` 出 IC 表 + 相关阵 + composite，
    落盘 JSON/CSV；`--synthetic` 离线自检。
  - 填补 `evaluate_all` 算出去相关子集却丢弃、`templates` 仅手工等权的空缺。
- **因子库 30 → 36**（commit `af69d5a`）：F031 52周高点邻近(George-Hwang 2004)、
  F032 月度季节性(Heston-Sadka 2008)、F033 下行 beta(Ang-Chen-Xing 2006)、
  F034 信息离散度 frog-in-pan(Da-Gurun-Warachka 2014)、F035 前景理论 TK
  价值(Barberis et al. 2016)、F036 近端最大回撤反转。均纯 OHLCV、与最近既有
  因子 `|corr|<0.4`。
- **股东户数真实数据层**（roadmap DS3/UB2，commit `a20c55e`）：
  `data/shareholders.py` —— akshare 股东户数 provider（明确英文 schema）+
  增量 per-symbol parquet 持久化 + `attach_to_panel()` as-of 合并（严格无前视），
  使 F003 股东集中度用真实户数而非换手代理。

### 变更 (Changed)
- `compute_all_factors(panel, log_errors=False)`：新增 opt-in `log_errors`
  参数暴露失败因子；默认行为不变（commit `af69d5a`）。
- `FACTOR_LIBRARY.json` v1.0 → v1.1（36 因子）。

### 仍待处理 (Top pending)
- **WS-A 可信度基建**（UA1 PIT-universe / UA2 ADV-impact / UD1 risk-overlay）
  已在 `data/` 实现，但**仅接入 `p0_engine_v2.py`，未接入 canonical
  `engine.py`/`run.py`** —— 仍是"让数字变真"的第一优先缺口。

## 2026-07-01 — Phase 2 Tier 4：canonical 引擎 opt-in 可信度切片

### 新增 (Added)
- **PIT 无偏 universe 开关**（roadmap UA1，commit `3046230`）：
  `engine.py` 的 `run()` 新增两个**向后兼容**参数（默认关 = 字节级复现原行为，
  零回归）：
  - `pit_universe: bool = False` —— True 时每个调仓日剔除"截至该日已退市"标的，
    委托 `data/delisted.py` 的 `DelistManager.get_delisted_before(date)`；
    惰性构造，仅 opt-in 时才 import 退市模块 / 触网。
  - `delist_manager: DelistManager | None = None` —— 可注入（测试/自定义缓存）。
  - 诚实边界：仅能剔除 panel 中**已含**的已退市标的；完整无偏还需数据层纳入退市
    标的（与 `p0_engine_v2.py` 现有实现一致）。
  - 测试 `test_engine_pit_universe.py`（10 例，合成 + 临时退市 CSV，全离线）。
  - 回归：`test_engine_extras(20)`/`test_engine_fast(5, 实盘数据)`/
    `test_ic_engine(3)`/`test_limits(2)` 全绿；`engine.py` diff = +23 加法行。

### 仍待处理 (Top pending)
- **UA2 ADV 平方根冲击成本** + **UD1 风险护栏** 仍未接入 canonical 引擎
  （固定 0.2% 滑点仍在用）——下一个可信度切片（`data/adv_impact.py` /
  `data/risk_overlay.py` 已就绪，需接入 `run()` 成本循环 / 仓位构建）。
