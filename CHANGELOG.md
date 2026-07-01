# Changelog

本文件记录 backtest_mvp 的高层变更。因子级细节见 `FACTOR_LIBRARY.json`，
alpha 路线图进度见 `roadmap_alpha_uplift.json` 的 `progress_log`。

## 2026-07-01 — Phase 4b：新增 F043 价格延迟（roadmap WS-C UC7 补齐）

### 新增 (Added)
- **F043 价格延迟** `price_delay`（Hou-Moskowitz 2005, RFS，commit `7849047` 因子+测试、
  JSON v1.3）——补齐 Phase 4 延后项，**独立的信息扩散 family**：
  - `delay = 1 − R²_restricted / R²_full`，限制模型 `r_i = a + b0·r_m`，
    完整模型 `r_i = a + b0·r_m + Σ_{k=1..4} b_k·r_m(−k)`，市场代理=截面均值收益。
  - 逻辑：价格对市场信息滞后响应（高 delay）的股票关注度低、扩散慢，获得溢价。
  - 实现：逐 symbol 滚动嵌套 OLS（numpy lstsq），min_periods 护栏；面向研究/中等
    universe（全量 survivorship 热循环慎用）。
  - `FACTOR_REGISTRY` 42→43；`FACTOR_LIBRARY.json` v1.2→v1.3。
  - 测试 `test_factors_p5.py` +4 例（29 全绿）：滞后响应股 delay 排名更高（构造性单调，
    LAGGED 1.000 > CONTEMP 0.900）、无前视、与 F025 去相关、集成。
  - **实测去相关最佳**：|corr| vs F025/F034/F016/F023 均 <0.03（确为独立 family）。

## 2026-07-01 — Phase 4：新增去相关 Alpha 因子 F037-F042（roadmap WS-C UC7）

### 新增 (Added)
- **6 个去相关 Alpha 因子**（roadmap WS-C UC7，commit `4fb5540` 因子+测试、
  `db347d1` FACTOR_LIBRARY.json v1.2）。从近期文献取灵感，各来自不同 family
  以最小化互相关，纯用现有 panel 列、有出处、加法式追加（零回归）：
  - **F037 系统性协偏度** `coskewness`（Harvey-Siddique 2000, JF）——
    `-rank(E[εᵢ·ε_m²]/(std(εᵢ)·var(ε_m)))`，系统性三阶共矩；区别于 F022 自身偏度。
  - **F038 隔夜-日内拉锯** `overnight_intraday_tug`（Lou-Polk-Skouras 2019, JFE）——
    21d 累积(隔夜−日内)收益；区别于 F012 单日跳空。
  - **F039 换手率变异系数** `turnover_cv`（Chordia 等 2001, JFE）——
    `std/mean` 流动性二阶矩；区别于 F018 换手水平。
  - **F040 隔夜跳空方差占比** `overnight_variance_share`（Parkinson 1980）——
    `隔夜var/(隔夜var+日内Parkinson var)`，**首个使用 high-low 区间的因子**，
    尺度无关比值与波动水平去相关（区别于 F006）。
  - **F041 水下时间/回撤持续期** `time_under_water`（路径依赖风险）——
    高水位下方时长占比；区别于 F036 回撤幅度。
  - **F042 非流动性变化** `delta_amihud`（Amihud 2002）——
    近段−前段 Amihud 变化率；区别于 F011 Amihud 水平。
  - `FACTOR_REGISTRY` 36→42；`FACTOR_LIBRARY.json` v1.1→v1.2（补 5 篇 source_papers、
    P4_academic/P5_decorrelated 分组，并补齐历史遗漏的 F031-F036 分组）。
  - 测试 `test_factors_p5.py`（25 例，合成快测）：对齐/短历史安全、每因子构造性
    单调 sanity、无前视、与最近邻既有因子去相关、`compute_all_factors` 集成。

### 去相关口径 (Note)
- repo 单测约定 `|corr|<0.8`（全部通过）；**设计目标 `|corr|<0.4` 针对真实截面数据**。
  合成随机面板（纯噪声 + 共同市场因子）为去相关最坏情形，F040/F041 在其上与最近邻
  相关 ~0.5（区间比值 / 回撤族在随机游走上人为相关），实盘截面数据预期显著更低。

### 仍待处理 (Top pending)
- **F043 价格延迟**（Hou-Moskowitz 2005）已在 Phase 4b 补齐（见上）。
- 接入 `combiner` 对 43 因子实测 IC / 去相关合成；接入 `overfitting.py` 出 PBO/DSR 部署裁决。

## 2026-07-01 — Phase 3：严谨过拟合诊断（roadmap UA3）

### 新增 (Added)
- **严谨多重检验 / 回测过拟合统计**（roadmap UA3，commit `1bd523f`）：
  新建 `research_loop/overfitting.py`，取代 `deflated_sharpe.py` 的粗糙 haircut
  近似（`DSR≈SR−ln(N)/(4√T)`，其 significance 分布假设不成立）。**不改旧模块**
  （`p0_engine_v2.py` 仍可用），结构性零回归。
  - `probabilistic_sharpe_ratio` —— 含偏度/峰度修正的 PSR
    `= Φ[(SR−SR*)·√(n−1) / √(1−γ3·SR+(γ4−1)/4·SR²)]`。
  - `expected_maximum_sharpe` —— Bailey–López de Prado 极值近似
    `E[max_N SR] = √Var·[(1−γ)Φ⁻¹(1−1/N)+γΦ⁻¹(1−1/(Ne))]`（N=1 护栏）。
  - `deflated_sharpe_ratio` —— 真 DSR = `PSR(SR*=E[max])`；
    `deflated_sharpe_from_trials` 便捷入口（自动取 best/N/Var）。
  - `probability_of_backtest_overfitting` —— **PBO via CSCV**（组合对称交叉验证）：
    IS 最优策略的 OOS 相对秩 → logit → `PBO = P(λ≤0)`，附性能衰减斜率 /
    P(OOS<0) / OOS 秩中位数。
  - `overfitting_report` + `__main__` 离线自检。
  - 测试 `test_overfitting.py`（27 例，全合成确定性）：PSR 单调性/基准=0.5/
    偏度峰度符号；E[max] 随 N 增长与 N=1/零方差边界；DSR 去膨胀逻辑；
    PBO（占优≈0.09 / 16-seed 均值≈0.48 / 构造过拟合=1.0 / `C(S,S/2)` / 输入校验）。

### 仍待处理 (Top pending)
- 把 `overfitting.py` 接入因子 zoo 评价器，对 36 因子 + 策略变体**实测** PBO/DSR，
  产出诚实的"可否部署"裁决（backlog）。

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
