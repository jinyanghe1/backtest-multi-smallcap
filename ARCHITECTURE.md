# backtest_mvp 架构说明书 & 接口说明书 & 优化方案

> 粒度：极细（行号级引用 + 具体代码片段 + 数据源链路）
> 参考框架：WorldQuant Alpha Research (wq-alpha-research) + 101 Formulaic Alphas
> 生成时间：2026-06-24

---

## 一、整体定位

**项目定位**：A股微盘股截面量化回测系统 → **升级为具备自我进化能力的因子研究平台**

**升级目标**：
- 因子逻辑层：高胜率模板体系（参照 WorldQuant `group_rank + ts_rank` 范式）
- 算子优化层：标准化算子库（delay / delta / ts_rank / decay_linear / correlation）
- 研究循环框架：IDEATE → SIMULATE → VALIDATE → CORRELATE → SUBMIT → MONITOR → EVOLVE
- 数据字段层：财务因子体系 + 行业分类（申万/Wind/GICS）

---

## 二、目录结构（重构后）

```
backtest_mvp/
├── engine.py                     # 核心回测引擎 [现状保留]
│
├── factors/                     # 【新建】因子体系
│   ├── __init__.py
│   ├── operators.py             # 算子定义（WQT风格）
│   │   ├── ts_rank(series, window)
│   │   ├── delay(series, periods)
│   │   ├── delta(series, periods)
│   │   ├── decay_linear(series, window)
│   │   ├── group_rank(series, group)
│   │   ├── correlation(x, y, window)
│   │   ├── rank(series)
│   │   ├── scale(series)
│   │   ├── ts_argmax(series, window)
│   │   ├── ts_argmin(series, window)
│   │   └── signed_power(series, alpha)
│   │
│   ├── financial_fields.py       # 【新建】财务因子字段定义
│   │   ├── FUNDAMENTAL_FIELDS   # 盈利能力/成长/负债/营运
│   │   ├── VALUATION_FIELDS     # PE/PB/PS/PCF/EV/EBITDA
│   │   ├── MARKET_FIELDS        # 市值/股本/股东户数
│   │   └── DERIVED_FIELDS       # 衍生比率（roe_growth_ttm, etc）
│   │
│   └── templates.py              # 【新建】高胜率模板库
│       ├── template_fundamental_value()   # Template A
│       ├── template_analyst_estimate()    # Template B
│       ├── template_technical_momentum()   # Template C
│       ├── template_multi_factor_blend()   # Template D
│       └── GOLDEN_COMBO                  # group_rank(ts_rank(signal,N), subindustry)
│
├── factors.py                   # 【现状→待废弃】旧因子计算模块
│                                 # 重构后入口迁移至 factors/operators.py
│
├── strategies.py                # 策略库 S1-S11 [现状保留]
├── strategies_v2.py             # 策略库 v2 [现状保留]
│
├── industry/                     # 【新建】行业分类
│   ├── __init__.py
│   ├── classifier.py            # 行业分类获取
│   │   ├── get_shenwan_industry(symbol) → str   # 申万一级
│   │   ├── get_shenwan_industry_2(symbol) → str  # 申万二级
│   │   ├── get_wind_industry(symbol) → str       # Wind一级
│   │   ├── get_citics_industry(symbol) → str    # 中信一级
│   │   └── get_gics_subindustry(symbol) → str   # GICS子行业
│   └── shenwan_map.parquet       # 申万行业成分股映射
│
├── data/                        # 【重构】数据获取层
│   ├── providers.py              # 【新建】统一数据获取接口
│   │   ├── class DataProvider
│   │   │   ├── PROVIDERS = {...}   # 各category的优先级
│   │   │   ├── get(category, symbol, field, **kwargs) → pd.Series
│   │   │   ├── _fetch(provider, category, symbol, field) → pd.Series
│   │   │   └── _validate(data) → bool
│   │   │
│   │   └── class ProviderError / NetworkError / RateLimitError
│   │
│   ├── field_resolver.py         # 【新建】字段冲突解决
│   │   ├── PRIORITY_RULES       # {field: [(source, priority), ...]}
│   │   ├── resolve(field, sources) → pd.Series
│   │   └── detect_conflict(sources, threshold=0.01) → bool
│   │
│   ├── source_eastmoney.py       # 【新建】东方财富数据源
│   │   ├── fetch_financials(symbol, report_type) → DataFrame
│   │   ├── fetch_daily_mcap_pb(symbol) → DataFrame
│   │   └── fetch_industry_class() → DataFrame
│   │
│   ├── source_ths.py             # 【新建】同花顺数据源
│   │   ├── fetch_financials(symbol) → DataFrame
│   │   └── fetch_industry_class() → DataFrame
│   │
│   ├── source_westock.py         # 【重构】westock数据源
│   │   ├── fetch_kline(symbol, days) → DataFrame
│   │   ├── fetch_quote(symbols) → DataFrame
│   │   └── fetch_microcap_universe() → DataFrame
│   │
│   ├── source_akshare.py         # 【重构】akshare数据源
│   │   ├── fetch_financials_cninfo(symbol) → DataFrame
│   │   ├── fetch_profile_cninfo(symbol) → DataFrame
│   │   ├── fetch_sw_industry() → DataFrame
│   │   └── fetch_wind_industry() → DataFrame
│   │
│   ├── source_sina.py            # 【新建】新浪数据源
│   │   └── fetch_financials_vip(symbol) → DataFrame
│   │
│   └── westock_data_script/      # westock node.js 脚本（本地）
│       └── index.js
│
├── research_loop/                # 【新建】研究循环框架
│   ├── __init__.py
│   ├── loop.py                   # 研究循环引擎
│   │   ├── class AlphaCandidate
│   │   │   ├── expr: str              # 因子表达式
│   │   │   ├── sharpe: float
│   │   │   ├── fitness: float
│   │   │   ├── turnover: float
│   │   │   ├── drawdown: float
│   │   │   ├── self_correlation: float
│   │   │   └── status: str            # ACTIVE/INACTIVE/RETIRED
│   │   │
│   │   └── class ResearchLoop
│   │       ├── design_alpha(idea: str) → AlphaCandidate
│   │       ├── simulate(alpha: AlphaCandidate) → BacktestResult
│   │       ├── validate_metrics(result) → bool
│   │       ├── check_self_correlation(alpha) → float
│   │       ├── submit(alpha: AlphaCandidate) → bool
│   │       ├── monitor(alpha: AlphaCandidate) → dict
│   │       └── evolve(alpha: AlphaCandidate, feedback: dict)
│   │
│   ├── evolve.py                 # 自进化机制
│   │   ├── class KnowledgeBase
│   │   │   ├── lessons: List[Lesson]
│   │   │   ├── add_lesson(lesson: Lesson)
│   │   │   ├── query(pattern: str) → List[Lesson]
│   │   │   └── export() → dict
│   │   │
│   │   └── class Lesson
│   │       ├── lesson_id: str
│   │       ├── pattern: str           # 因子表达式模式
│   │       ├── metrics: dict          # {sharpe, turnover, fitness}
│   │       ├── context: str            # 使用场景描述
│   │       └── created_at: datetime
│   │
│   └── validators.py              # 指标校验
│       ├── METRIC_THRESHOLDS = {
│       │     'sharpe': 1.5,    # ≥1.5 (最低1.25)
│       │     'fitness': 1.0,    # Sharpe × √(Returns/TO)
│       │     'turnover': (0.01, 0.20),  # 1%-20%
│       │     'drawdown': 0.15,  # <15%
│       │     'self_correlation': 0.7   # <0.7
│       │ }
│       └── validate(alpha_result) → ValidationResult
│
├── benchmark.py                  # 基准指数对比 [现状保留]
├── analyze.py                   # IC分析/因子审计 [现状保留]
├── data.py                      # 数据下载 [现状→待废弃，重构至data/]
├── fetch_financials.py         # 财务季报采集 [现状→待废弃]
├── fetch_missing_financials.py # 缺失财务补全 [现状→待废弃]
├── update_daily.py             # 日线增量更新 [现状保留]
├── run.py                       # 主入口/CLI [现状保留]
├── quick_backtest.py            # 快速回测脚本 [现状保留]
│
├── tests/                       # 【扩展】测试
│   ├── test_benchmark.py
│   ├── test_limits.py
│   ├── test_ic_weights.py
│   ├── test_s7_s11.py
│   ├── test_operators.py        # 【新建】算子单元测试
│   ├── test_financial_fields.py # 【新建】财务因子测试
│   ├── test_field_resolver.py  # 【新建】字段冲突解决测试
│   ├── test_research_loop.py    # 【新建】研究循环测试
│   └── test_industry.py        # 【新建】行业分类测试
│
├── configs/                     # 【新建】配置
│   ├── default.yaml             # 默认回测配置
│   ├── providers.yaml           # 数据源优先级配置
│   └── metric_thresholds.yaml   # 指标阈值配置
│
├── caches/                      # 缓存目录
│   ├── data_cache/              # A股日线
│   ├── financials_cache/        # 财务季报
│   ├── profiles_cache/         # 总股本
│   ├── daily_mcap_pb_cache/    # 逐日mcap/pb/pe
│   ├── industry_cache/         # 【新建】行业分类缓存
│   ├── alpha_base/             # 【新建】因子知识库
│   └── name_lookup.parquet
│
├── benchmarks/                  # 基准指数
├── DATA.md                      # 数据集定义文档
└── pipeline.sh                   # 全量采集流水线
```

---

## 三、数据流架构（5层）

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 0: 外部数据源                                                 │
│  ─────────────────────────────────────────────────────────────────  │
│  东方财富 (eastmoney)     ← 财务季报、公告日、市值、行业的首选源      │
│  同花顺 (ths)             ← 财务数据 fallback #1                      │
│  新浪 (sina)              ← 财务数据 fallback #2                    │
│  AkShare (akshare)        ← 行业分类（申万/ Wind / 中信）           │
│  Westock (westock-data)   ← 日线 OHLCV（本地 node.js）              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ data/providers.py: DataProvider.get()
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: 字段解析与冲突解决层                                       │
│  ─────────────────────────────────────────────────────────────────  │
│  data/field_resolver.py                                              │
│    ├── PRIORITY_RULES: {field: [(source, priority), ...]}            │
│    ├── resolve(field, sources) → 优先级合并                           │
│    └── detect_conflict(sources) → 差异超阈值则告警                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ data/industry/classifier.py
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2: 行业分类层                                                 │
│  ─────────────────────────────────────────────────────────────────  │
│  industry/classifier.py                                              │
│    ├── get_shenwan_industry(symbol)  → 申万一级（30个行业）           │
│    ├── get_shenwan_industry_2(symbol) → 申万二级（100+个行业）        │
│    ├── get_wind_industry(symbol)     → Wind一级                       │
│    ├── get_citics_industry(symbol)  → 中信一级                        │
│    └── get_gics_subindustry(symbol) → GICS子行业（77个）              │
│                                                                     │
│  用途：group_rank 中性化、申万行业分组回测、板块轮动因子               │
└────────────────────────────┬────────────────────────────────────────┘
                             │ factors/operators.py: 算子计算
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 3: 因子逻辑层（高胜率模板）                                    │
│  ─────────────────────────────────────────────────────────────────  │
│  factors/templates.py                                                │
│    ├── Template A: group_rank(ts_rank(field/equity, 126), subindustry)│
│    │               → 基础价值因子，预期 Sharpe≥1.5, TO=2-8%          │
│    ├── Template B: group_rank(ts_rank(est_eps/close, 126), industry) │
│    │               → 分析师预期因子，预期 Sharpe≥1.5, TO=9-16%        │
│    ├── Template C: decay_linear(ts_rank(delta(close,1),8), decay)   │
│    │               → 技术动量因子，预期 TO=15-35%                    │
│    ├── Template D: equal_weight(f1, f2, f3)                         │
│    │               → 多因子混合，预期降低相关性                        │
│    └── GOLDEN_COMBO: group_rank(ts_rank(signal,N), subindustry)     │
│                       → WQ验证的黄金组合                             │
│                                                                     │
│  factors/financial_fields.py                                         │
│    ├── FUNDAMENTAL: roe, roa, gross_margin, net_margin               │
│    ├── GROWTH: revenue_growth_ttm, profit_growth_ttm, eps_growth     │
│    ├── VALUATION: pe_ttm, pb, ps, pcf, ev/ebitda                    │
│    └── MARKET: float_mcap, total_mcap, float_shares, total_shares    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ engine.py: CrossSectionalEngine.run()
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 4: 回测执行层                                                 │
│  ─────────────────────────────────────────────────────────────────  │
│  BacktestResult: equity_curve, annual_return, sharpe, max_drawdown  │
│                  calmar_ratio, win_rate, avg_turnover, positions_log│
└────────────────────────────┬────────────────────────────────────────┘
                             │ research_loop/loop.py: ResearchLoop
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5: 研究循环与进化层                                           │
│  ─────────────────────────────────────────────────────────────────  │
│  research_loop/loop.py: 7阶段循环                                    │
│    IDEATE → SIMULATE → VALIDATE → CORRELATE → SUBMIT →             │
│    MONITOR → EVOLVE                                                  │
│                                                                     │
│  research_loop/evolve.py:                                           │
│    ├── KnowledgeBase: lessons（因子模式库）                            │
│    └── Lesson: {pattern, metrics, context, created_at}               │
│                                                                     │
│  research_loop/validators.py:                                       │
│    ├── METRIC_THRESHOLDS                                            │
│    └── validate(alpha_result) → ValidationResult                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、模块依赖关系（精确）

```
┌─────────────────────────────────────────────────────────┐
│  入口层: run.py / quick_backtest.py                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  研究循环层: research_loop/loop.py                       │
│    ├── ResearchLoop.run(idea)                            │
│    ├── design_alpha() → 调用 factors/templates.py        │
│    ├── simulate() → 调用 engine.py                       │
│    ├── check_self_correlation() → 调用 factors/operators│
│    └── evolve() → 调用 research_loop/evolve.py          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  因子逻辑层: factors/templates.py + factors/operators.py │
│    ├── templates.py: 高胜率模板（调用 operators）         │
│    └── operators.py: ts_rank/delay/delta/correlation... │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  数据层: data/providers.py + data/field_resolver.py      │
│    ├── providers.py: 统一获取（调用各 source_*.py）       │
│    ├── field_resolver.py: 冲突解决                       │
│    ├── source_eastmoney.py                               │
│    ├── source_ths.py                                     │
│    ├── source_sina.py                                   │
│    ├── source_akshare.py                                 │
│    ├── source_westock.py                                 │
│    └── industry/classifier.py                           │
└─────────────────────────────────────────────────────────┘
```

---

## 五、算子优化层（factors/operators.py）

### 5.1 算子定义（WorldQuant 风格）

所有算子签名统一为：

```python
# factors/operators.py

def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """
    时间序列排名算子
    含义：series 在过去 window 天的截面排名百分位（0~1）
    公式：ts_rank(x, 10) = rank(x[-10:]) 的最后一个值
    用途：标准化时间序列中的相对位置，消除量纲影响
    """
    return series.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)


def delay(series: pd.Series, periods: int) -> pd.Series:
    """
    延迟算子
    公式：delay(x, 1) = x[t-1]
    """
    return series.shift(periods)


def delta(series: pd.Series, periods: int) -> pd.Series:
    """
    差分算子
    公式：delta(x, 1) = x[t] - x[t-1]
    """
    return series.diff(periods)


def decay_linear(series: pd.Series, window: int) -> pd.Series:
    """
    线性衰减加权算子
    公式：decay_linear(x, n)[t] = (x[t] * n + x[t-1] * (n-1) + ... + x[t-n+1] * 1) / sum(1..n)
    用途：近期信号权重更高，平滑噪声
    """
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=False
    )


def group_rank(series: pd.Series, group: pd.Series) -> pd.Series:
    """
    组内排名算子
    公式：group_rank(x, industry) = rank(x) within each industry group
    用途：行业中性化，消除行业间系统性差异
    实现：
        df = pd.DataFrame({'value': series, 'group': group})
        return df.groupby('group')['value'].rank(pct=True)
    """
    df = pd.DataFrame({'value': series, 'group': group})
    return df.groupby('group')['value'].rank(pct=True)


def correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    滚动相关系数算子
    公式：rolling_corr(x, y, window)
    用途：量价背离、因子相关性分析
    """
    return x.rolling(window).corr(y)


def rank(series: pd.Series) -> pd.Series:
    """
    截面排名算子
    公式：rank(x) = x 的截面百分位排名（0~1）
    用途：跨截面可比性，消除极端值影响
    """
    return series.rank(pct=True)


def scale(series: pd.Series) -> pd.Series:
    """
    标准化算子（A股专用）
    公式：scale(x) = x / sum(|x|)
    用途：保持方向性，但使因子值可跨截面累加
    注意：A股不允许做空，scale后权重恒正
    """
    return series / series.abs().sum()


def ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """
    时间序列窗口内最大值的位置
    用途：捕捉近期极值点，用于事件驱动因子
    """
    return series.rolling(window).apply(lambda x: x.argmax(), raw=False)


def ts_argmin(series: pd.Series, window: int) -> pd.Series:
    """
    时间序列窗口内最小值的位置
    """
    return series.rolling(window).apply(lambda x: x.argmin(), raw=False)


def signed_power(series: pd.Series, alpha: float) -> pd.Series:
    """
    带符号的幂算子
    公式：signed_power(x, a) = sign(x) * |x|^a
    用途：放大/压缩极端值，如 signed_power(delta(close,1), 2)
    """
    return np.sign(series) * (series.abs() ** alpha)
```

### 5.2 算子参数设置表（按因子类型）

| 因子类型 | decay | neutralization | nan_handling | 预期 TO | 预期 Sharpe |
|---------|-------|----------------|--------------|---------|-----------|
| **基础价值** | 0 | SUBINDUSTRY | ON | 2-8% | ≥1.5 |
| **成长能力** | 0 | INDUSTRY | ON | 5-12% | ≥1.3 |
| **分析师预期** | 0-4 | INDUSTRY/SUBINDUSTRY | ON | 9-16% | ≥1.5 |
| **技术动量** | 10-30 | INDUSTRY | OFF | 15-35% | ≥1.0 |
| **情绪/新闻** | 4-10 | INDUSTRY | ON | 8-30% | ≥0.8 |
| **低波动** | 0 | SUBINDUSTRY | ON | 2-10% | ≥1.2 |

### 5.3 nan_handling 策略

```python
# factors/operators.py

def nan_safe_rank(series: pd.Series, min_periods_ratio: float = 0.5) -> pd.Series:
    """
    处理 NaN 的安全排名
    window 内有效值 < min_periods_ratio * window 时返回 NaN
    """
    def _safe_rank(x):
        valid = x.dropna()
        if len(valid) < len(x) * min_periods_ratio:
            return np.nan
        return valid.rank(pct=True).iloc[-1]
    return series.rolling(20).apply(_safe_rank, raw=False)
```

---

## 六、因子逻辑层（factors/templates.py）

### 6.1 高胜率模板库

```python
# factors/templates.py

"""
WorldQuant 验证的高胜率因子模板
来源：wq-alpha-research SKILL.md + 101 Formulaic Alphas
"""


# ─────────────────────────────────────────────────────────────────
# Template A: 基础价值因子
# group_rank(ts_rank(field/equity, 126), subindustry)
# 预期 Sharpe≥1.5, Turnover=2-8%, 适合熊市防御
# ─────────────────────────────────────────────────────────────────

def template_fundamental_value(
    snapshot: pd.DataFrame,
    field: str = 'roe_ttm',
    window: int = 126,
    neutralization: str = 'subindustry'
) -> pd.Series:
    """
    基础价值因子模板

    参数：
        snapshot: 因子截面，包含 field 和中性化分组
        field: 财务字段名，支持 roe_ttm, roa_ttm, gross_margin, net_margin
        window: ts_rank 窗口，默认 126（约半年交易日）
        neutralization: 中性化分组，默认 subindustry（申万二级）

    示例：
        template_fundamental_value(snapshot, 'roe_ttm', 126, 'subindustry')
        → group_rank(ts_rank(roe_ttm, 126), subindustry)

    数据获取：
        roe_ttm → 来自 financials_cache，计算式：net_profit / total_equity
        subindustry → 来自 industry_cache/申万二级行业
    """
    # Step 1: 计算 ts_rank(field, window)
    ts = ts_rank(snapshot[field], window)

    # Step 2: 组内排名（中性化）
    if neutralization == 'subindustry':
        group = snapshot['sw_industry_2']
    elif neutralization == 'industry':
        group = snapshot['sw_industry_1']
    elif neutralization == 'wind':
        group = snapshot['wind_industry']
    elif neutralization == 'citics':
        group = snapshot['citics_industry']
    else:
        group = None

    if group is not None:
        return group_rank(ts, group)
    else:
        return rank(ts)


# ─────────────────────────────────────────────────────────────────
# Template B: 分析师预期因子
# group_rank(ts_rank(est_eps/close, 126), industry)
# 预期 Sharpe≥1.5, Turnover=9-16%, 适合趋势市
# ─────────────────────────────────────────────────────────────────

def template_analyst_estimate(
    snapshot: pd.DataFrame,
    field: str = 'est_eps',
    window: int = 126,
    neutralization: str = 'industry'
) -> pd.Series:
    """
    分析师预期因子模板

    参数：
        field: 分析师预期字段，支持 est_eps（预期EPS）, est_revenue, est_profit
        注意：est_* 字段需要数据源支持，暂无完整A股分析师预期覆盖

    备选方案（无分析师数据时）：
        用 growth_ttm（净利润增速）替代 est_eps
        → group_rank(ts_rank(growth_ttm/close, 126), industry)
    """
    signal = snapshot[field] / snapshot['close']
    ts = ts_rank(signal, window)
    return group_rank(ts, snapshot['industry'])


# ─────────────────────────────────────────────────────────────────
# Template C: 技术动量因子（带 decay 平滑）
# decay_linear(ts_rank(delta(close, 1), 8), decay)
# 预期 TO=15-35%, 适合短线交易
# ─────────────────────────────────────────────────────────────────

def template_technical_momentum(
    snapshot: pd.DataFrame,
    price_field: str = 'close',
    delta_periods: int = 1,
    ts_rank_window: int = 8,
    decay_window: int = 20,
    neutralization: str = 'industry'
) -> pd.Series:
    """
    技术动量因子模板

    公式：decay_linear(ts_rank(delta(close, 1), 8), 20)

    逻辑：
        delta(close, 1) → 日收益率
        ts_rank(..., 8) → 过去8天收益排名（相对位置）
        decay_linear(..., 20) → 线性加权平滑，降低换手

    调参建议：
        decay_window: 10-30, 越大换手越低但信号越滞后
        delta_periods: 1=日频, 5=周频
    """
    delta_val = delta(snapshot[price_field], delta_periods)
    ts = ts_rank(delta_val, ts_rank_window)
    decayed = decay_linear(ts, decay_window)
    return group_rank(decayed, snapshot[neutralization])


# ─────────────────────────────────────────────────────────────────
# Template D: 多因子等权混合
# 用途：降低单一因子波动，提升稳定性
# ─────────────────────────────────────────────────────────────────

def template_multi_factor_blend(
    snapshot: pd.DataFrame,
    signals: list,
    weights: list = None,
    neutralization: str = 'subindustry'
) -> pd.Series:
    """
    多因子混合模板

    参数：
        signals: 因子 Series 的列表，如 [signal1, signal2, signal3]
        weights: 权重列表，默认 None = 等权
        neutralization: 中性化分组

    公式：blend = sum(w_i * rank(signal_i)) / sum(w)

    示例：
        s1 = template_fundamental_value(snapshot, 'roe_ttm')
        s2 = template_fundamental_value(snapshot, 'gross_margin')
        s3 = template_technical_momentum(snapshot, 'close')
        blend = template_multi_factor_blend(snapshot, [s1, s2, s3], weights=[0.4, 0.3, 0.3])
    """
    ranks = [rank(s) for s in signals]

    if weights is None:
        blended = sum(ranks) / len(ranks)
    else:
        blended = sum(w * r for w, r in zip(weights, ranks))

    if neutralization:
        return group_rank(blended, snapshot[neutralization])
    return blended


# ─────────────────────────────────────────────────────────────────
# GOLDEN_COMBO: WQ 验证的黄金组合
# group_rank(ts_rank(signal, N), subindustry)
# 核心思想：在 subindustry 组内做 ts_rank，等效行业中性 + 趋势提取
# ─────────────────────────────────────────────────────────────────

GOLDEN_COMBO_SIGNALS = [
    'roe_ttm',
    'revenue_growth_ttm',
    'gross_margin',
    'mom20d',
    'vol20d',       # 低波动因子（负向）
]

def golden_combo(snapshot: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    黄金组合因子

    逻辑：
        1. 对每个信号计算 ts_rank(signal, n)
        2. 等权相加
        3. 在 subindustry 内做 group_rank

    参数：
        n: ts_rank 窗口，默认 20

    预期效果：
        - 多信号混合降低个别因子失效风险
        - subindustry 中性化消除行业轮动影响
        - Sharpe 预期比单因子高 20-30%
    """
    ts_ranks = []
    for field in GOLDEN_COMBO_SIGNALS:
        if field in snapshot.columns:
            if field == 'vol20d':
                # 低波动因子：越小越好，取负
                ts_ranks.append(-ts_rank(snapshot[field], n))
            else:
                ts_ranks.append(ts_rank(snapshot[field], n))

    combined = sum(ts_ranks) / len(ts_ranks)
    return group_rank(combined, snapshot['sw_industry_2'])
```

---

## 七、财务因子字段定义（factors/financial_fields.py）

### 7.1 字段分类体系

```python
# factors/financial_fields.py

"""
A股财务因子字段定义
数据源优先级：eastmoney > ths > sina > akshare
字段冲突解决：见 data/field_resolver.py
"""

# ─────────────────────────────────────────────────────────────────
# 基础字段（直接来自原始财务数据）
# ─────────────────────────────────────────────────────────────────

RAW_FINANCIAL_FIELDS = {
    # ── 资产负债表 ──
    'total_assets': {
        'name': '资产总计',
        'unit': '元',
        'source': ['eastmoney', 'ths', 'sina'],
        'alias': ['total_assets', 'ASSETS', 'ZONC'],
    },
    'total_liabilities': {
        'name': '负债合计',
        'unit': '元',
        'source': ['eastmoney', 'ths', 'sina'],
    },
    'total_equity': {
        'name': '股东权益合计',
        'unit': '元',
        'source': ['eastmoney', 'ths', 'sina'],
        'alias': ['shareholders_equity', 'GQJZ'],
    },
    'net_assets': {
        'name': '净资产',
        'unit': '元',
        'source': ['eastmoney', 'ths'],
    },
    'current_assets': {
        'name': '流动资产',
        'unit': '元',
        'source': ['eastmoney', 'ths'],
    },
    'non_current_assets': {
        'name': '非流动资产',
        'unit': '元',
        'source': ['eastmoney', 'ths'],
    },

    # ── 利润表 ──
    'revenue': {
        'name': '营业总收入',
        'unit': '元',
        'source': ['eastmoney', 'ths', 'sina'],
        'alias': ['operating_revenue', 'YYSR'],
    },
    'operating_income': {
        'name': '营业收入',
        'unit': '元',
        'source': ['eastmoney', 'ths', 'sina'],
    },
    'net_profit': {
        'name': '归属净利润',
        'unit': '元',
        'source': ['eastmoney', 'ths', 'sina'],
        'alias': ['parent_net_profit', 'JLR'],
    },
    'gross_profit': {
        'name': '毛利润',
        'unit': '元',
        'source': ['eastmoney', 'ths'],
    },
    'operating_profit': {
        'name': '营业利润',
        'unit': '元',
        'source': ['eastmoney', 'ths'],
    },
    'ebit': {
        'name': '息税前利润',
        'unit': '元',
        'source': ['eastmoney', 'ths'],
    },

    # ── 现金流量表 ──
    'operating_cash_flow': {
        'name': '经营活动现金流',
        'unit': '元',
        'source': ['eastmoney', 'ths', 'sina'],
    },
    'investing_cash_flow': {
        'name': '投资活动现金流',
        'unit': '元',
        'source': ['eastmoney', 'ths'],
    },
    'financing_cash_flow': {
        'name': '筹资活动现金流',
        'unit': '元',
        'source': ['eastmoney', 'ths'],
    },

    # ── 每股数据 ──
    'eps': {
        'name': '基本每股收益',
        'unit': '元/股',
        'source': ['eastmoney', 'ths', 'sina'],
        'alias': ['eps_basic', 'MGJYX'],
    },
    'diluted_eps': {
        'name': '稀释每股收益',
        'unit': '元/股',
        'source': ['eastmoney', 'ths'],
    },
    'bps': {
        'name': '每股净资产',
        'unit': '元/股',
        'source': ['eastmoney', 'ths', 'sina'],
        'alias': ['book_value_per_share', 'MGJZC'],
    },
    'cps': {
        'name': '每股经营现金流',
        'unit': '元/股',
        'source': ['eastmoney', 'ths'],
    },

    # ── 分红数据 ──
    'dividend_per_share': {
        'name': '每股股息',
        'unit': '元/股',
        'source': ['eastmoney', 'ths'],
    },
    'dividend_yield': {
        'name': '股息率(%)',
        'unit': '%',
        'source': ['eastmoney', 'ths'],
    },
}

# ─────────────────────────────────────────────────────────────────
# 衍生比率字段（计算得出）
# ─────────────────────────────────────────────────────────────────

DERIVED_RATIO_FIELDS = {
    # ── 盈利能力 ──
    'roe_ttm': {
        'name': 'ROE(TTM)',
        'formula': 'net_profit_ttm / total_equity',
        'unit': 'ratio',
        'literature': 'Fama-French 1993',
    },
    'roa_ttm': {
        'name': 'ROA(TTM)',
        'formula': 'net_profit_ttm / total_assets',
        'unit': 'ratio',
    },
    'gross_margin': {
        'name': '毛利率',
        'formula': 'gross_profit / revenue',
        'unit': 'ratio',
    },
    'net_margin': {
        'name': '净利率',
        'formula': 'net_profit / revenue',
        'unit': 'ratio',
    },
    'operating_margin': {
        'name': '营业利润率',
        'formula': 'operating_profit / revenue',
        'unit': 'ratio',
    },
    'ebit_margin': {
        'name': 'EBIT利润率',
        'formula': 'ebit / revenue',
        'unit': 'ratio',
    },

    # ── 成长能力 ──
    'revenue_growth_yoy': {
        'name': '营收同比增速',
        'formula': '(revenue - revenue_lag4Q) / revenue_lag4Q',
        'unit': 'ratio',
        'note': '同比4季度前',
    },
    'profit_growth_yoy': {
        'name': '净利润同比增速',
        'formula': '(net_profit - net_profit_lag4Q) / net_profit_lag4Q',
        'unit': 'ratio',
    },
    'revenue_growth_ttm': {
        'name': '营收TTM增速',
        'formula': '(revenue_ttm - revenue_ttm_lag4Q) / revenue_ttm_lag4Q',
        'unit': 'ratio',
    },
    'profit_growth_ttm': {
        'name': '净利润TTM增速',
        'formula': '(net_profit_ttm - net_profit_ttm_lag4Q) / net_profit_ttm_lag4Q',
        'unit': 'ratio',
    },
    'eps_growth_yoy': {
        'name': 'EPS同比增速',
        'formula': '(eps - eps_lag4Q) / eps_lag4Q',
        'unit': 'ratio',
    },

    # ── 负债水平 ──
    'debt_ratio': {
        'name': '资产负债率',
        'formula': 'total_liabilities / total_assets',
        'unit': 'ratio',
    },
    'current_ratio': {
        'name': '流动比率',
        'formula': 'current_assets / current_liabilities',
        'unit': 'ratio',
    },
    'quick_ratio': {
        'name': '速动比率',
        'formula': '(current_assets - inventory) / current_liabilities',
        'unit': 'ratio',
    },

    # ── 营运能力 ──
    'asset_turnover': {
        'name': '资产周转率',
        'formula': 'revenue / total_assets',
        'unit': 'ratio',
    },
    'inventory_turnover': {
        'name': '存货周转率',
        'formula': 'operating_cost / average_inventory',
        'unit': 'ratio',
    },
    'receivables_turnover': {
        'name': '应收账款周转率',
        'formula': 'revenue / average_receivables',
        'unit': 'ratio',
    },

    # ── 估值指标 ──
    'pe_ttm': {
        'name': '市盈率(TTM)',
        'formula': 'close / eps_ttm',
        'unit': 'ratio',
        'alias': ['pe', 'P/E'],
    },
    'pb': {
        'name': '市净率',
        'formula': 'close / bps',
        'unit': 'ratio',
        'alias': ['P/B', 'pb_ratio'],
    },
    'ps': {
        'name': '市销率',
        'formula': 'close * total_shares / revenue',
        'unit': 'ratio',
    },
    'pcf': {
        'name': 'PCF比率',
        'formula': 'close * total_shares / operating_cash_flow',
        'unit': 'ratio',
    },
    'ev_ebitda': {
        'name': 'EV/EBITDA',
        'formula': 'enterprise_value / ebitda_ttm',
        'unit': 'ratio',
    },

    # ── 股息相关 ──
    'dividend_payout_ratio': {
        'name': '分红率',
        'formula': 'dividend_per_share / eps',
        'unit': 'ratio',
    },
}

# ─────────────────────────────────────────────────────────────────
# 市场数据字段
# ─────────────────────────────────────────────────────────────────

MARKET_FIELDS = {
    'float_mcap': {
        'name': '流通市值',
        'unit': '亿元',
        'source': ['westock', 'eastmoney'],
        'formula': 'close * float_shares',
    },
    'total_mcap': {
        'name': '总市值',
        'unit': '亿元',
        'source': ['westock', 'eastmoney'],
        'formula': 'close * total_shares',
    },
    'float_shares': {
        'name': '流通股本',
        'unit': '亿股',
        'source': ['akshare', 'eastmoney'],
    },
    'total_shares': {
        'name': '总股本',
        'unit': '亿股',
        'source': ['akshare', 'eastmoney'],
    },
    'free_shares': {
        'name': '自由流通股本',
        'unit': '亿股',
        'source': ['westock'],
    },
    'close': {
        'name': '收盘价(前复权)',
        'unit': '元',
        'source': ['westock', 'akshare'],
    },
    'volume': {
        'name': '成交量',
        'unit': '股',
        'source': ['westock', 'akshare'],
    },
    'turnover_rate': {
        'name': '换手率(%)',
        'unit': '%',
        'source': ['westock', 'akshare'],
        'formula': 'volume / float_shares * 100',
    },
    'adv20': {
        'name': '20日平均成交额',
        'unit': '元',
        'formula': 'ma(volume*close, 20)',
    },
}

# ─────────────────────────────────────────────────────────────────
# 字段汇总
# ─────────────────────────────────────────────────────────────────

ALL_FIELDS = {
    **RAW_FINANCIAL_FIELDS,
    **DERIVED_RATIO_FIELDS,
    **MARKET_FIELDS,
}
```

---

## 八、行业分类层（industry/classifier.py）

### 8.1 行业分类体系

```python
# industry/classifier.py

"""
A股行业分类接口
支持：申万（SW）、Wind、中信（Citics）、GICS
"""

from enum import Enum

class IndustrySource(Enum):
    SHENWAN_1 = 'sw_industry_1'   # 申万一级（30个）
    SHENWAN_2 = 'sw_industry_2'   # 申万二级（100+个）
    SHENWAN_3 = 'sw_industry_3'   # 申万三级（300+个）
    WIND_1 = 'wind_industry_1'    # Wind一级
    WIND_2 = 'wind_industry_2'    # Wind二级
    CITICS_1 = 'citics_industry_1'  # 中信一级
    CITICS_2 = 'citics_industry_2'  # 中信二级
    GICS = 'gics_subindustry'     # GICS子行业（77个）


SHENWAN_LEVEL_1 = [
    '银行', '非银金融', '房地产', '建筑材料', '建筑装饰',
    '钢铁', '有色金属', '煤炭', '石油石化', '基础化工',
    '轻工制造', '纺织服装', '家用电器', '食品饮料', '农林牧渔',
    '医药生物', '汽车', '汽车零部件', '电力设备', '机械设备',
    '电子', '计算机', '传媒', '通信', '国防军工',
    '商贸零售', '社会服务', '交通运输', '公用事业', '综合',
]

INDUSTRY_DATA_SOURCES = {
    'sw_industry': {
        'source': 'akshare',
        'api': 'akshare.sw_index_weight(stock_code)',  # 通过指数成分获取行业
        'fallback': 'akshare.sw_industry_cons()',
        'citics_code': 'sh000001',  # 申万一级指数代码
        'update_freq': '季度',
    },
    'wind_industry': {
        'source': 'akshare',
        'api': 'akshare.wind_industry_class()',
        'fallback': None,
        'update_freq': '年度',
    },
    'citics_industry': {
        'source': 'akshare',
        'api': 'akshare.zt_index_cons(quote_symbol)',  # 中信指数成分
        'fallback': None,
        'update_freq': '年度',
    },
    'gics_subindustry': {
        'source': 'eastmoney',
        'api': 'eastmoney.gics_classification()',
        'fallback': None,
        'note': 'GICS全球行业分类标准，77个子行业',
    },
}
```

### 8.2 行业分类获取接口

```python
# industry/classifier.py

def get_sw_industry_1(symbol: str) -> str:
    """
    获取申万一级行业分类

    接口：
        akshare.sw_index_weight(symbol) → 获取该股票所在的申万行业指数

    Fallback 链：
        1. akshare.sw_index_weight(symbol)
        2. akshare.sw_industry_cons() → 遍历查找
        3. eastmoney.sw_classification(symbol)

    异常处理：
        - SymbolNotFound: 股票代码无效
        - IndustryDataMissing: 行业数据缺失，尝试 wind_industry
    """
    try:
        # 方法1: 直接通过股票代码查申万行业指数
        df = akshare.sw_index_weight(symbol=symbol)
        if not df.empty:
            return df['index_name'].iloc[0]
    except (APIError, NetworkError):
        pass

    try:
        # 方法2: 遍历申万一级指数成分
        for idx_code in SW_1_INDEX_CODES:
            df = akshare.sw_index_cons(index_code=idx_code)
            if symbol in df['symbol'].values:
                return SW_1_NAME_MAP[idx_code]
    except (APIError, NetworkError):
        pass

    # 方法3: Fallback 到 Wind
    return get_wind_industry_1(symbol)


def get_sw_industry_2(symbol: str) -> str:
    """
    获取申万二级行业分类
    用于 group_rank(signal, subindustry) 中性化
    """
    # 实现同 get_sw_industry_1，但查询二级指数
    ...


def get_gics_subindustry(symbol: str) -> str:
    """
    获取 GICS 子行业分类

    用于国际化可比的中性化分组

    数据源：
        eastmoney: gics_subindustry（精确到77个子行业）
        akshare: gics_classification

    异常处理：
        - GICSDataUnavailable: 该股票无 GICS 分类（如港股）
        → Fallback 到 sw_industry_2
    """
    try:
        df = akshare.gics_classification(symbol=symbol)
        if not df.empty:
            return df['subindustry'].iloc[0]
    except (APIError, NetworkError):
        pass

    # Fallback
    return get_sw_industry_2(symbol)
```

---

## 九、数据获取接口层（data/providers.py）

### 9.1 统一数据获取接口

```python
# data/providers.py

"""
统一数据获取接口
支持多数据源 fallback，自动重试，异常处理
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 数据源优先级配置
# ─────────────────────────────────────────────────────────────────

PROVIDERS = {
    'price': {
        'primary': 'westock',
        'fallback': ['akshare', 'sina'],
        'timeout': 30,
    },
    'financials': {
        'primary': 'eastmoney',
        'fallback': ['ths', 'sina'],
        'timeout': 60,
    },
    'fundamental': {
        'primary': 'akshare',
        'fallback': ['eastmoney', 'cninfo'],
        'timeout': 60,
    },
    'industry': {
        'primary': 'akshare',
        'fallback': ['eastmoney', 'ths'],
        'timeout': 30,
    },
    'quote': {
        'primary': 'westock',
        'fallback': ['akshare', 'sina'],
        'timeout': 10,
    },
}


# ─────────────────────────────────────────────────────────────────
# 异常定义
# ─────────────────────────────────────────────────────────────────

class ProviderError(Exception):
    """数据源错误基类"""
    pass


class NetworkError(ProviderError):
    """网络相关错误，可重试"""
    pass


class RateLimitError(ProviderError):
    """限流错误，等待后重试"""
    pass


class FieldNotFoundError(ProviderError):
    """字段不存在，需切换数据源"""
    pass


class DataValidationError(ProviderError):
    """数据校验失败，不重试"""
    pass


class AllProvidersFailedError(ProviderError):
    """所有数据源均失败"""
    def __init__(self, category, symbol, field, errors):
        self.category = category
        self.symbol = symbol
        self.field = field
        self.errors = errors  # {provider: error_message}
        super().__init__(
            f"All providers failed for {category}/{symbol}/{field}: {errors}"
        )


# ─────────────────────────────────────────────────────────────────
# 数据获取类
# ─────────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    data: pd.Series
    source: str
    timestamp: pd.Timestamp
    is_complete: bool  # 是否完整数据（非空、非全Na）


class DataProvider:
    """
    统一数据获取接口

    用法：
        provider = DataProvider()
        series = provider.get('financials', '000001', 'roe_ttm')
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or PROVIDERS
        self._init_providers()

    def _init_providers(self):
        """初始化各数据源客户端"""
        from data.source_westock import WestockClient
        from data.source_eastmoney import EastMoneyClient
        from data.source_ths import THSClient
        from data.source_akshare import AkshareClient
        from data.source_sina import SinaClient

        self.clients = {
            'westock': WestockClient(),
            'eastmoney': EastMoneyClient(),
            'ths': THSClient(),
            'akshare': AkshareClient(),
            'sina': SinaClient(),
        }

    def get(
        self,
        category: str,
        symbol: str,
        field: str,
        **kwargs
    ) -> FetchResult:
        """
        统一获取接口

        参数：
            category: 数据类别（price/financials/fundamental/industry/quote）
            symbol: 股票代码
            field: 字段名
            **kwargs: 额外参数（如 report_date, period 等）

        流程：
            1. 按 PROVIDERS[category] 顺序尝试各数据源
            2. 每个数据源最多重试 3 次
            3. 成功则校验数据有效性
            4. 全部失败则抛出 AllProvidersFailedError

        异常处理：
            NetworkError → 等待3s重试，最多3次
            RateLimitError → 等待60s重试
            FieldNotFoundError → 立即切换下一个数据源
            DataValidationError → 记录日志，切换数据源
        """
        providers = self.config[category]
        errors = {}

        for provider_name in [providers['primary']] + providers['fallback']:
            client = self.clients[provider_name]

            for attempt in range(3):
                try:
                    data = self._fetch(
                        client, provider_name, category, symbol, field, **kwargs
                    )
                    if self._validate(data):
                        return FetchResult(
                            data=data,
                            source=provider_name,
                            timestamp=pd.Timestamp.now(),
                            is_complete=True,
                        )
                    else:
                        raise DataValidationError(
                            f"Data validation failed for {field}"
                        )

                except NetworkError as e:
                    logger.warning(
                        f"[{provider_name}] NetworkError attempt {attempt+1}: {e}"
                    )
                    if attempt < 2:
                        time.sleep(3 * (2 ** attempt))  # 3, 6, 12s
                        continue
                    errors[provider_name] = str(e)

                except RateLimitError as e:
                    logger.warning(
                        f"[{provider_name}] RateLimitError: {e}"
                    )
                    time.sleep(60)
                    continue

                except FieldNotFoundError as e:
                    logger.warning(
                        f"[{provider_name}] FieldNotFound: {field}"
                    )
                    errors[provider_name] = str(e)
                    break  # 切换下一个数据源

                except DataValidationError as e:
                    logger.error(f"[{provider_name}] ValidationError: {e}")
                    errors[provider_name] = str(e)
                    break  # 切换下一个数据源

                except Exception as e:
                    logger.error(
                        f"[{provider_name}] Unexpected error: {e}"
                    )
                    errors[provider_name] = str(e)
                    break

        raise AllProvidersFailedError(category, symbol, field, errors)

    def _fetch(
        self,
        client,
        provider: str,
        category: str,
        symbol: str,
        field: str,
        **kwargs
    ) -> pd.Series:
        """
        调用具体数据源客户端
        """
        if provider == 'westock':
            if category == 'price':
                return client.fetch_kline(symbol, **kwargs)['close']
            elif category == 'quote':
                return client.fetch_quote([symbol])[symbol]

        elif provider == 'eastmoney':
            if category == 'financials':
                return client.fetch_financials(symbol, field, **kwargs)
            elif category == 'fundamental':
                return client.fetch_fundamental(symbol, field, **kwargs)
            elif category == 'industry':
                return client.fetch_industry(symbol, **kwargs)

        elif provider == 'ths':
            ...

        elif provider == 'akshare':
            ...

        elif provider == 'sina':
            ...

        raise FieldNotFoundError(f"Unknown category/field: {category}/{field}")

    def _validate(self, data: pd.Series) -> bool:
        """
        数据校验
        满足以下全部条件返回 True：
        1. 非空
        2. 非全 NaN
        3. 非全 0（对于有实际值的字段）
        4. 日期连续（可配置）
        """
        if data is None or data.empty:
            return False
        if data.isna().all():
            return False
        # 注意：0 在财务数据中可能是有效值（如 eps=0 表示亏损）
        # 所以这里不检查全0
        return True
```

### 9.2 各数据源接口详情

```python
# data/source_eastmoney.py

"""
东方财富数据源
"""

class EastMoneyClient:
    """
    东方财富（EastMoney）数据接口

    数据覆盖：
        - 财务季报（report_date, notice_date, bps, eps, revenue, net_profit）
        - 逐日市值（mcap, pb, pe）
        - 行业分类

    优势：
        - 公告日精确
        - 数据完整性高
        - 财务字段丰富

    官网：https://www.eastmoney.com
    """

    BASE_URL = 'https://push2.eastmoney.com'

    def fetch_financials(
        self,
        symbol: str,
        field: str,
        report_type: str = 'annual'
    ) -> pd.Series:
        """
        获取财务数据

        接口：
            EM 财务报告 API: /api/f10/llfp
            参数：secid, report_type, begin_date, end_date

        返回：
            pd.Series: index=report_date, values=field值

        异常：
            - NetworkError: 网络超时
            - FieldNotFoundError: 字段不存在
            - RateLimitError: 限流
        """
        url = f"{self.BASE_URL}/api/f10/llfp"
        params = {
            'secid': self._symbol_to_em_code(symbol),
            'report_type': report_type,
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            raise RateLimitError("EastMoney rate limit")
        if resp.status_code != 200:
            raise NetworkError(f"HTTP {resp.status_code}")

        data = resp.json()
        if 'data' not in data or not data['data']:
            raise FieldNotFoundError(f"Field {field} not found")

        df = pd.DataFrame(data['data'])
        series = df.set_index('report_date')[field]
        return series


# data/source_ths.py

class THSClient:
    """
    同花顺数据源

    数据覆盖：
        - 财务数据（bps, eps, revenue, net_profit）
        - 行业分类

    优势：
        - 行业分类准确
        - 财务报表科目名称规范

    官网：https://www.10jqka.com.cn
    """

    def fetch_financials(self, symbol: str, field: str) -> pd.Series:
        """
        接口：THS 财务数据 API
        Fallback: 同花顺有时延，建议作为 EM 的备用
        """
        ...


# data/source_akshare.py

class AkshareClient:
    """
    AkShare 数据源（综合）

    数据覆盖：
        - cninfo: 巨潮财务数据（总股本等）
        - sw_index: 申万行业指数成分
        - wind_industry: Wind行业分类

    优势：
        - 接口统一
        - 数据源广泛

    官网：https://www.akshare.xyz
    """

    def fetch_sw_industry(self, symbol: str) -> str:
        """
        获取申万行业分类

        接口：
            akshare.sw_index_weight(secid=sh000001)
            通过遍历申万一级指数成分确定股票行业
        """
        ...

    def fetch_profile_cninfo(self, symbol: str) -> pd.Series:
        """
        获取总股本（来自 cninfo）

        接口：
            akshare.stock_profile_cninfo(symbol='000001')
            → 注册资金（万元）→ / 10000 = 亿股
        """
        ...
```

---

## 十、字段冲突解决（data/field_resolver.py）

### 10.1 优先级规则

```python
# data/field_resolver.py

"""
多数据源字段冲突解决策略
"""

PRIORITY_RULES = {
    # ── 每股数据 ──
    'bps': [
        ('eastmoney', 1),   # 东方财富（公告日精确）
        ('ths', 2),         # 同花顺
        ('sina', 3),        # 新浪
    ],
    'eps': [
        ('eastmoney', 1),
        ('ths', 2),
        ('sina', 3),
    ],

    # ── 利润表 ──
    'revenue': [
        ('eastmoney', 1),
        ('ths', 2),
        ('sina', 3),
    ],
    'net_profit': [
        ('eastmoney', 1),
        ('ths', 2),
        ('sina', 3),
    ],

    # ── 市值相关 ──
    'mcap': [
        ('westock', 1),     # 前复权价格计算
        ('eastmoney', 2),
        ('akshare', 3),
    ],
    'float_mcap': [
        ('westock', 1),
        ('eastmoney', 2),
        ('akshare', 3),
    ],
    'close': [
        ('westock', 1),     # 前复权日线
        ('akshare', 2),      # 后复权
        ('sina', 3),         # 不复权
    ],

    # ── 股本相关 ──
    'total_shares': [
        ('akshare', 1),      # cninfo 精确
        ('eastmoney', 2),
    ],
    'float_shares': [
        ('westock', 1),     # 自由流通比例
        ('akshare', 2),
    ],

    # ── 行业分类 ──
    'sw_industry_1': [
        ('akshare', 1),     # sw_index_weight
        ('eastmoney', 2),   # EM 申万分类
        ('ths', 3),
    ],
    'sw_industry_2': [
        ('akshare', 1),
        ('eastmoney', 2),
        ('ths', 3),
    ],
    'wind_industry': [
        ('akshare', 1),     # wind_industry_class
        ('eastmoney', 2),
    ],
}


class FieldResolver:
    """
    字段冲突解决器

    用法：
        resolver = FieldResolver()
        merged = resolver.resolve('bps', {
            'eastmoney': series_em,
            'ths': series_ths,
            'sina': series_sina,
        })
    """

    def resolve(self, field: str, sources: Dict[str, pd.Series]) -> pd.Series:
        """
        按优先级合并多数据源字段

        参数：
            field: 字段名
            sources: {source_name: series}，必须包含 index=date

        返回：
            合并后的 pd.Series

        策略：
            1. 有 PRIORITY_RULES[field] → 按优先级取第一个非空
            2. 无 PRIORITY_RULES → 取第一个非空
            3. 全部为空 → 抛出异常
        """
        if not sources:
            raise ValueError(f"No sources provided for field {field}")

        # 去重：同一 source 可能返回重复日期
        deduped = {}
        for src, series in sources.items():
            if series is not None and not series.empty:
                deduped[src] = series[~series.index.duplicated(keep='last')]

        if not deduped:
            raise ValueError(f"All sources empty for field {field}")

        if field in PRIORITY_RULES:
            priority_list = PRIORITY_RULES[field]
            for source, _ in priority_list:
                if source in deduped:
                    s = deduped[source]
                    if not s.isna().all():
                        return s

        # 无优先级规则：取第一个非空
        for src, series in deduped.items():
            if not series.isna().all():
                return series

        raise ValueError(f"All sources invalid for field {field}")

    def detect_conflict(
        self,
        sources: Dict[str, pd.Series],
        threshold: float = 0.01
    ) -> bool:
        """
        检测冲突：不同数据源差异是否超过阈值

        参数：
            sources: {source_name: series}
            threshold: 相关性阈值，默认0.01（即相关性 < 0.99 视为冲突）

        返回：
            True = 存在冲突，需要人工检查
            False = 无冲突
        """
        if len(sources) < 2:
            return False

        series_list = [
            s for s in sources.values()
            if s is not None and not s.empty and not s.isna().all()
        ]

        if len(series_list) < 2:
            return False

        # 两两检查相关性
        for i in range(len(series_list)):
            for j in range(i + 1, len(series_list)):
                # 对齐日期
                aligned = series_list[i].align(series_list[j], join='inner')
                corr = aligned[0].corr(aligned[1])
                if abs(corr) < (1 - threshold):
                    logger.warning(
                        f"Conflict detected: sources[{i}] vs sources[{j}] "
                        f"corr={corr:.4f} < {1-threshold:.4f}"
                    )
                    return True

        return False
```

---

## 十一、研究循环框架（research_loop/loop.py）

### 11.1 研究循环 7 阶段

```python
# research_loop/loop.py

"""
研究循环引擎
参照 wq-alpha-research 的 IDEATE → SIMULATE → VALIDATE → CORRELATE → SUBMIT → MONITOR → EVOLVE
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd


@dataclass
class AlphaCandidate:
    """因子候选"""
    expr: str                              # 因子表达式
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    drawdown: float = 0.0
    self_correlation: float = 0.0          # 与活跃因子相关度
    status: str = 'DRAFT'                  # DRAFT/ACTIVE/INACTIVE/RETIRED
    created_at: datetime = field(default_factory=datetime.now)
    backtest_result: Optional[BacktestResult] = None


class ResearchLoop:
    """
    因子研究循环引擎

    7 阶段流程：
        1. IDEATE: 自然语言想法 → 因子表达式
        2. SIMULATE: 本地回测
        3. VALIDATE: 指标校验
        4. CORRELATE: 自相关检查
        5. SUBMIT: 提交模拟/实盘
        6. MONITOR: 监控表现
        7. EVOLVE: 自我进化
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.engine = CrossSectionalEngine(...)
        self.knowledge_base = KnowledgeBase()
        self.validator = MetricsValidator()
        self.alpha_db: Dict[str, AlphaCandidate] = {}

    def run(self, idea: str) -> AlphaCandidate:
        """
        完整研究循环
        """
        # Stage 1: IDEATE
        alpha = self.design_alpha(idea)

        # Stage 2: SIMULATE
        result = self.simulate(alpha)

        # Stage 3: VALIDATE
        if not self.validate_metrics(result):
            alpha.status = 'REJECTED'
            return alpha

        # Stage 4: CORRELATE
        corr = self.check_self_correlation(alpha)
        alpha.self_correlation = corr

        if corr >= 0.7:
            # 相关性过高，需要修改
            alpha = self.modify_for_low_correlation(alpha)

        # Stage 5: SUBMIT
        self.submit(alpha)

        # Stage 6: MONITOR
        self.monitor(alpha)

        # Stage 7: EVOLVE
        self.evolve(alpha)

        return alpha

    def design_alpha(self, idea: str) -> AlphaCandidate:
        """
        Stage 1: IDEATE
        将自然语言想法转换为因子表达式

        实现方式：
            1. 解析 idea 中的关键词（价值/动量/低波动/成长）
            2. 匹配 templates.py 中的模板
            3. 组装因子表达式

        示例：
            idea="申万二级行业下 ROE 最高的股票"
            → template_fundamental_value(snapshot, 'roe_ttm', neutralization='subindustry')
        """
        keywords = self._extract_keywords(idea)
        expr = self._match_template(keywords)
        return AlphaCandidate(expr=expr)

    def simulate(self, alpha: AlphaCandidate) -> BacktestResult:
        """
        Stage 2: SIMULATE
        本地回测
        """
        # 解析 alpha.expr
        snapshot = self._prepare_snapshot(alpha.expr)

        # 调用 engine.run()
        result = self.engine.run(
            universe_filter=...,
            ranking_fn=self._expr_to_ranking_fn(alpha.expr),
        )

        alpha.backtest_result = result
        alpha.sharpe = result.sharpe_ratio
        alpha.fitness = result.sharpe_ratio * np.sqrt(result.annual_return / result.avg_turnover)
        alpha.turnover = result.avg_turnover
        alpha.drawdown = result.max_drawdown

        return result

    def validate_metrics(self, result: BacktestResult) -> bool:
        """
        Stage 3: VALIDATE
        指标校验

        阈值（来自 wq-alpha-research）：
            Sharpe ≥ 1.5（最低 1.25）
            Fitness = Sharpe × √(Returns/TO) ≥ 1.0
            Turnover 1%-20%
            Drawdown < 15%
        """
        return self.validator.validate(result)

    def check_self_correlation(self, alpha: AlphaCandidate) -> float:
        """
        Stage 4: CORRELATE
        计算与 ACTIVE 因子的日收益相关性

        相关性 < 0.7 才能提交
        ≥ 0.7 → 需要修改
        """
        alpha_returns = self._get_alpha_daily_returns(alpha.expr)

        active_alphas = [
            a for a in self.alpha_db.values()
            if a.status == 'ACTIVE'
        ]

        if not active_alphas:
            return 0.0

        correlations = []
        for active in active_alphas:
            active_returns = self._get_alpha_daily_returns(active.expr)
            aligned = alpha_returns.align(active_returns, join='inner')
            corr = aligned[0].corr(aligned[1])
            correlations.append(abs(corr))

        return max(correlations) if correlations else 0.0

    def submit(self, alpha: AlphaCandidate) -> bool:
        """
        Stage 5: SUBMIT
        提交模拟或实盘
        """
        alpha.status = 'ACTIVE'
        self.alpha_db[alpha.expr] = alpha
        return True

    def monitor(self, alpha: AlphaCandidate) -> dict:
        """
        Stage 6: MONITOR
        监控因子表现
        """
        # 定期计算 rolling Sharpe，更新 alpha 状态
        ...

    def evolve(self, alpha: AlphaCandidate) -> Lesson:
        """
        Stage 7: EVOLVE
        自我进化：将成功/失败经验加入知识库
        """
        if alpha.status == 'ACTIVE' and alpha.sharpe >= 1.5:
            lesson = Lesson(
                pattern=alpha.expr,
                metrics={
                    'sharpe': alpha.sharpe,
                    'turnover': alpha.turnover,
                    'fitness': alpha.fitness,
                },
                context=self._describe_context(alpha),
            )
            self.knowledge_base.add_lesson(lesson)
            return lesson
        return None
```

### 11.2 指标校验器（validators.py）

```python
# research_loop/validators.py

"""
指标校验器
阈值来自 wq-alpha-research 的实证研究（625个因子样本）
"""

METRIC_THRESHOLDS = {
    'sharpe': {
        'min': 1.25,
        'target': 1.5,
        'note': '最低1.25，目标1.5'
    },
    'fitness': {
        'min': 1.0,
        'target': 1.5,
        'formula': 'sharpe × √(returns / turnover)',
    },
    'turnover': {
        'min': 0.01,   # 1%
        'max': 0.20,  # 20%
        'note': '过低（<1%）可能过度集中，过高（>20%）交易成本大'
    },
    'drawdown': {
        'max': 0.15,  # 15%
        'note': '超过15%需要优化或放弃'
    },
    'self_correlation': {
        'max': 0.7,
        'note': '与现有因子相关性超过0.7需要修改'
    },
}


@dataclass
class ValidationResult:
    is_valid: bool
    failures: List[str]
    warnings: List[str]
    scores: Dict[str, float]


class MetricsValidator:
    """
    指标校验器
    """

    def validate(self, alpha: AlphaCandidate) -> ValidationResult:
        failures = []
        warnings = []

        # Sharpe
        if alpha.sharpe < METRIC_THRESHOLDS['sharpe']['min']:
            failures.append(
                f"Sharpe {alpha.sharpe:.2f} < {METRIC_THRESHOLDS['sharpe']['min']} (min)"
            )
        elif alpha.sharpe < METRIC_THRESHOLDS['sharpe']['target']:
            warnings.append(
                f"Sharpe {alpha.sharpe:.2f} < {METRIC_THRESHOLDS['sharpe']['target']} (target)"
            )

        # Fitness
        if alpha.fitness < METRIC_THRESHOLDS['fitness']['min']:
            failures.append(
                f"Fitness {alpha.fitness:.2f} < {METRIC_THRESHOLDS['fitness']['min']}"
            )

        # Turnover
        if alpha.turnover < METRIC_THRESHOLDS['turnover']['min']:
            warnings.append(
                f"Turnover {alpha.turnover:.2%} < {METRIC_THRESHOLDS['turnover']['min']:.2%} (too low)"
            )
        elif alpha.turnover > METRIC_THRESHOLDS['turnover']['max']:
            failures.append(
                f"Turnover {alpha.turnover:.2%} > {METRIC_THRESHOLDS['turnover']['max']:.2%} (too high)"
            )

        # Drawdown
        if alpha.drawdown > METRIC_THRESHOLDS['drawdown']['max']:
            warnings.append(
                f"Drawdown {alpha.drawdown:.2%} > {METRIC_THRESHOLDS['drawdown']['max']:.2%}"
            )

        return ValidationResult(
            is_valid=len(failures) == 0,
            failures=failures,
            warnings=warnings,
            scores={
                'sharpe': alpha.sharpe,
                'fitness': alpha.fitness,
                'turnover': alpha.turnover,
                'drawdown': alpha.drawdown,
            }
        )
```

### 11.3 自进化机制（evolve.py）

```python
# research_loop/evolve.py

"""
自进化机制
参照 wq-alpha-research evolve_skill.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import json
import pathlib


@dataclass
class Lesson:
    """
    因子模式的经验教训
    """
    lesson_id: str
    pattern: str                      # 因子表达式模式
    metrics: Dict[str, float]         # {sharpe, turnover, fitness}
    context: str                      # 使用场景描述
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


class KnowledgeBase:
    """
    因子知识库
    存储历史因子模式及其表现，用于后续搜索和复用
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or 'caches/alpha_base/lessons.jsonl'
        self.lessons: List[Lesson] = []
        self._load()

    def _load(self):
        """从磁盘加载知识库"""
        p = pathlib.Path(self.db_path)
        if p.exists():
            with open(p) as f:
                for line in f:
                    data = json.loads(line)
                    self.lessons.append(Lesson(**data))

    def _save(self):
        """持久化到磁盘"""
        p = pathlib.Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            for lesson in self.lessons:
                f.write(json.dumps(asdict(lesson), ensure_ascii=False) + '\n')

    def add_lesson(self, lesson: Lesson):
        """添加新经验"""
        self.lessons.append(lesson)
        self._save()

    def query(self, pattern: str = None, min_sharpe: float = None) -> List[Lesson]:
        """查询知识库"""
        results = self.lessons

        if pattern:
            results = [l for l in results if pattern in l.pattern]

        if min_sharpe is not None:
            results = [
                l for l in results
                if l.metrics.get('sharpe', 0) >= min_sharpe
            ]

        return results

    def export(self) -> Dict:
        """导出为 dict（供 agent 使用）"""
        return {
            'lessons': [
                {
                    'pattern': l.pattern,
                    'sharpe': l.metrics.get('sharpe'),
                    'turnover': l.metrics.get('turnover'),
                    'context': l.context,
                }
                for l in self.lessons[-20:]  # 最近20条
            ]
        }


def evolve_skill(
    new_alphas: List[AlphaCandidate],
    local_snapshot: List[AlphaCandidate],
    knowledge_base: KnowledgeBase
) -> List[Lesson]:
    """
    自进化主函数

    流程（来自 evolve_skill.py）：
        1. 获取用户新增/修改的因子列表
        2. 对比本地快照，找出差异
        3. 计算新因子与 ACTIVE 因子的日收益相关性
        4. 生成 Markdown lesson 片段
        5. 追加到知识库

    参数：
        new_alphas: 最新获取的因子列表
        local_snapshot: 上次保存的因子快照
        knowledge_base: 知识库实例

    返回：
        新增的 Lesson 列表
    """
    new_lessons = []

    # 找差异
    local_exprs = {a.expr for a in local_snapshot}
    for alpha in new_alphas:
        if alpha.expr not in local_exprs and alpha.status == 'ACTIVE':
            # 新增的活跃因子
            if alpha.sharpe >= 1.5:
                lesson = Lesson(
                    lesson_id=generate_lesson_id(),
                    pattern=alpha.expr,
                    metrics={
                        'sharpe': alpha.sharpe,
                        'turnover': alpha.turnover,
                        'fitness': alpha.fitness,
                    },
                    context=describe_alpha_context(alpha),
                )
                knowledge_base.add_lesson(lesson)
                new_lessons.append(lesson)

    return new_lessons
```

---

## 十二、核心接口说明书

### 12.1 factors/operators.py

| 接口 | 签名 | 说明 |
|------|------|------|
| `ts_rank` | `(series, window) → Series` | 时间序列排名 |
| `delay` | `(series, periods) → Series` | 延迟 |
| `delta` | `(series, periods) → Series` | 差分 |
| `decay_linear` | `(series, window) → Series` | 线性衰减加权 |
| `group_rank` | `(series, group) → Series` | 组内排名（中性化） |
| `correlation` | `(x, y, window) → Series` | 滚动相关系数 |
| `rank` | `(series) → Series` | 截面排名 |
| `scale` | `(series) → Series` | 标准化（A股用） |
| `ts_argmax` | `(series, window) → Series` | 窗口最大值位置 |
| `ts_argmin` | `(series, window) → Series` | 窗口最小值位置 |
| `signed_power` | `(series, alpha) → Series` | 带符号幂 |

### 12.2 factors/templates.py

| 接口 | 签名 | 说明 |
|------|------|------|
| `template_fundamental_value` | `(snapshot, field, window, neutralization) → Series` | Template A |
| `template_analyst_estimate` | `(snapshot, field, window, neutralization) → Series` | Template B |
| `template_technical_momentum` | `(snapshot, price_field, delta_periods, ts_rank_window, decay_window, neutralization) → Series` | Template C |
| `template_multi_factor_blend` | `(snapshot, signals, weights, neutralization) → Series` | Template D |
| `golden_combo` | `(snapshot, n) → Series` | 黄金组合 |

### 12.3 data/providers.py

| 接口 | 签名 | 说明 |
|------|------|------|
| `DataProvider.get` | `(category, symbol, field, **kwargs) → FetchResult` | 统一获取 |
| `DataProvider._fetch` | `(client, provider, category, symbol, field) → Series` | 调用数据源 |
| `DataProvider._validate` | `(data) → bool` | 数据校验 |

### 12.4 data/field_resolver.py

| 接口 | 签名 | 说明 |
|------|------|------|
| `FieldResolver.resolve` | `(field, sources) → Series` | 按优先级合并 |
| `FieldResolver.detect_conflict` | `(sources, threshold) → bool` | 检测冲突 |

### 12.5 industry/classifier.py

| 接口 | 签名 | 说明 |
|------|------|------|
| `get_sw_industry_1` | `(symbol) → str` | 申万一级行业 |
| `get_sw_industry_2` | `(symbol) → str` | 申万二级行业 |
| `get_wind_industry` | `(symbol) → str` | Wind行业 |
| `get_citics_industry` | `(symbol) → str` | 中信行业 |
| `get_gics_subindustry` | `(symbol) → str` | GICS子行业 |

### 12.6 research_loop/loop.py

| 接口 | 签名 | 说明 |
|------|------|------|
| `AlphaCandidate` | `@dataclass` | 因子候选数据结构 |
| `ResearchLoop.run` | `(idea) → AlphaCandidate` | 完整研究循环 |
| `ResearchLoop.design_alpha` | `(idea) → AlphaCandidate` | IDEATE 阶段 |
| `ResearchLoop.simulate` | `(alpha) → BacktestResult` | SIMULATE 阶段 |
| `ResearchLoop.validate_metrics` | `(result) → bool` | VALIDATE 阶段 |
| `ResearchLoop.check_self_correlation` | `(alpha) → float` | CORRELATE 阶段 |
| `ResearchLoop.submit` | `(alpha) → bool` | SUBMIT 阶段 |
| `ResearchLoop.monitor` | `(alpha) → dict` | MONITOR 阶段 |
| `ResearchLoop.evolve` | `(alpha) → Lesson` | EVOLVE 阶段 |

### 12.7 research_loop/evolve.py

| 接口 | 签名 | 说明 |
|------|------|------|
| `Lesson` | `@dataclass` | 经验数据结构 |
| `KnowledgeBase.add_lesson` | `(lesson)` | 添加经验 |
| `KnowledgeBase.query` | `(pattern, min_sharpe) → List[Lesson]` | 查询经验 |
| `evolve_skill` | `(new_alphas, local_snapshot, kb) → List[Lesson]` | 自进化主函数 |

---

## 十三、现有问题与优化方案

### P0 — 必须修复

| ID | 问题 | 修复方案 | 涉及文件 |
|----|------|---------|---------|
| P0-1 | `_ic_history` 全局状态污染 | 改用 `contextvars.ContextVar` | strategies.py |
| P0-2 | 跌停过滤缺失 | 补全 `is_limit_down` 过滤 | engine.py |

### P1 — 高优先级

| ID | 问题 | 修复方案 | 涉及文件 |
|----|------|---------|---------|
| P1-1 | 硬编码绝对路径 | 改为环境变量 + 相对路径 | data.py, quick_backtest.py |
| P1-2 | 佣金/滑点参数不统一 | 提取到 configs/default.yaml | engine.py, run.py |
| P1-3 | 测试覆盖率不足 | 新增 test_operators.py 等 | tests/ |

### P2 — 新增功能（WorldQuant 风格）

| ID | 功能 | 实现方案 | 涉及文件 |
|----|------|---------|---------|
| P2-1 | 算子层 | 新建 factors/operators.py | - |
| P2-2 | 高胜率模板 | 新建 factors/templates.py | - |
| P2-3 | 财务因子字段 | 新建 factors/financial_fields.py | - |
| P2-4 | 行业分类 | 新建 industry/classifier.py | - |
| P2-5 | 统一数据获取 | 新建 data/providers.py | - |
| P2-6 | 字段冲突解决 | 新建 data/field_resolver.py | - |
| P2-7 | 研究循环框架 | 新建 research_loop/ | - |
| P2-8 | 自进化机制 | 新建 research_loop/evolve.py | - |

---

## 十四、实现路径（Phase by Phase）

### Phase 1: 数据层重构（1-2周）
1. 新建 `data/providers.py` 统一获取接口
2. 新建 `data/field_resolver.py` 冲突解决
3. 新建 `data/source_*.py` 各数据源客户端
4. 新建 `industry/classifier.py` 行业分类
5. 迁移旧 `data.py` / `fetch_financials.py` 功能

### Phase 2: 因子层构建（2-3周）
1. 新建 `factors/operators.py` 算子库
2. 新建 `factors/financial_fields.py` 字段定义
3. 新建 `factors/templates.py` 高胜率模板
4. 废弃旧 `factors.py`（保留兼容）

### Phase 3: 研究循环（2-3周）
1. 新建 `research_loop/validators.py` 指标校验
2. 新建 `research_loop/loop.py` 研究引擎
3. 新建 `research_loop/evolve.py` 自进化
4. 新建 `configs/*.yaml` 配置文件

### Phase 4: 测试与迭代（1-2周）
1. 新增各模块单元测试
2. 端到端研究循环测试
3. 性能优化

---

## 附录 A: WQ 101 Alphas 算子速查

| 算子 | WQ公式 | 本框架实现 |
|------|--------|---------|
| ts_rank | `ts_rank(x, n)` | `ts_rank(series, n)` |
| delay | `delay(x, d)` | `delay(series, d)` |
| delta | `delta(x, d)` | `delta(series, d)` |
| decay_linear | `decay_linear(x, n)` | `decay_linear(series, n)` |
| correlation | `correlation(x, y, n)` | `correlation(x, y, n)` |
| rank | `rank(x)` | `rank(series)` |
| scale | `scale(x)` | `scale(series)` |
| group_rank | `group_rank(x, g)` | `group_rank(series, group)` |
| signed_power | `signed_power(x, a)` | `signed_power(series, a)` |
| sum | `sum(x, n)` | `series.rolling(n).sum()` |
| ts_max | `ts_max(x, n)` | `series.rolling(n).max()` |
| ts_min | `ts_min(x, n)` | `series.rolling(n).min()` |

## 附录 B: 申万一级行业（30个）

银行、非银金融、房地产、建筑材料、建筑装饰、钢铁、有色金属、煤炭、石油石化、基础化工、轻工制造、纺织服装、家用电器、食品饮料、农林牧渔、医药生物、汽车、汽车零部件、电力设备、机械设备、电子、计算机、传媒、通信、国防军工、商贸零售、社会服务、交通运输、公用事业、综合

## 附录 C: 数据源官网

| 数据源 | 官网 | 主要用途 |
|--------|------|---------|
| 东方财富 | https://www.eastmoney.com | 财务季报、公告日、市值 |
| 同花顺 | https://www.10jqka.com.cn | 财务 fallback、行业分类 |
| 新浪 | https://finance.sina.com.cn | 财务 fallback |
| AkShare | https://www.akshare.xyz | 行业分类（申万/Wind/中信）|
| Westock | 本地 node.js | 日线 OHLCV（前复权）|
