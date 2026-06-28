# Backtest Engine 优化框架与 Roadmap

> 作者: AI
> 日期: 2026-06-28
> 版本: v2.0-optim
> 目标: 将全量回测(5,400+股票)从>10min优化到<2min

---

## 1. 现状分析

### 1.1 瓶颈定位

通过 `cProfile` 和逐行分析，当前引擎在 5,400+ 股票上的瓶颈集中在 **DataFrame 索引查找**：

| 瓶颈点 | 调用频率 | 每次耗时 | 时间占比 | 根因 |
|-------|---------|---------|---------|------|
| `_get_factor_snapshot()` | ~60次/策略 × 调仓日 | 2-5ms | ~15% | MultiIndex `.xs()` 哈希查找 |
| `_get_daily_return()` | ~30只 × 20日 × 60月 = 36,000次 | 1-3ms | **~40%** | 每次 `.xs()` 查找 + `.reindex()` |
| `_get_period_returns()` | ~60次/策略 | 5-10ms | ~15% | `IndexSlice` 多行多列查找 |
| `_compute_ic_series()` | 每策略1次 | 全量循环 | **~20%** | 重复调用 snapshot + period_returns |
| `_compute_quantile_spread()` | 每策略1次 | 全量循环 | ~5% | 同上 |
| 中性化 OLS 回归 | ~60次 | 5-15ms | ~5% | `np.linalg.lstsq` 在循环中 |

**500只股票**: 回测 1.3s ✅  
**5,400只股票**: 回测 >10min ❌  
**时间差距**: 不是因为股票数量差10倍，而是索引查找 overhead 在数据量大时急剧恶化。

### 1.2 核心问题

`MultiIndex DataFrame` 的 `.xs()` 和 `.loc[IndexSlice]` 在大循环内被重复调用：

```python
# 每次调用都是一次哈希查找
r = self.returns.xs(ts, level=0, drop_level=True)  # O(n) in worst case
r = r.reindex(stocks)                               # O(n) 对齐
```

当 `self.returns` 有 700万行 (date, stock)，每次 `xs()` 需要遍历整个索引找到所有匹配行，这种 **O(N) 每次 × 万次循环** 的复杂度是性能杀手。

---

## 2. 优化总览

### 2.1 优化层级

```
┌─────────────────────────────────────────────────────────────┐
│  L3: 策略层面 (向量化)                                      │
│  ├─ 一次选出所有调仓日所有股票，而非逐日循环                │
│  └─ 矩阵乘法替代逐日迭代                                    │
├─────────────────────────────────────────────────────────────┤
│  L2: 数据层面 (预计算)                                      │
│  ├─ MultiIndex → 2D Pivot 矩阵 (date × stock)             │
│  ├─ 预计算累积收益率矩阵 cum_returns                       │
│  └─ 预计算滚动因子矩阵 (中性化)                            │
├─────────────────────────────────────────────────────────────┤
│  L1: 索引层面 (消除查找)                                    │
│  ├─ 日期→位置映射字典 (date → int)                         │
│  ├─ 股票→列号映射字典 (stock → int)                        │
│  └─ 直接用 numpy 数组索引替代 pandas.loc[]                  │
├─────────────────────────────────────────────────────────────┤
│  L0: 架构层面 (异步)                                        │
│  ├─ 多策略并行 (multiprocessing)                            │
│  └─ 多策略共享预处理数据                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 预期收益

| 优化项 | 当前耗时 | 目标耗时 | 加速比 |
|-------|---------|---------|-------|
| L1: 索引消除 | 50% | 5% | **10x** |
| L2: 数据预 pivot | 每次30ms | 每次0.3ms | **100x** |
| L3: 向量化回测循环 | 日频循环 | 矩阵乘法 | **50x** |
| L0: 多策略并行 | 串行 | 4核并行 | **4x** |
| **综合** | **>10min** | **<1.5min** | **~100x** |

---

## 3. 详细优化方案

### 3.1 方案 A: 预 Pivot 为 2D 矩阵 (L1 + L2)

**目标**: 将 `MultiIndex` 数据转换为 `(date, stock)` 的二维矩阵，用 `numpy` 索引。

**实现**:

```python
class CrossSectionalEngine:
    def __init__(self, factor_panel, return_panel, ...):
        # ... existing code ...
        
        # L1: 预构建映射
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.stock_to_idx = {s: i for i, s in enumerate(self.stocks)}
        
        # L2: 预 pivot 收益率矩阵
        self.returns_2d = self._pivot_returns(return_panel)
        self.factors_2d = self._pivot_factors(factor_panel)
        
        # L2: 预计算累积收益率
        self.cum_returns_2d = np.cumprod(1 + self.returns_2d, axis=0)
        self.log_returns_2d = np.log1p(self.returns_2d)  # 用于 log-return 求和
    
    def _pivot_returns(self, return_panel):
        """将 MultiIndex 收益率面板转为 2D numpy 数组 (date × stock)"""
        pivot = return_panel['daily_return'].unstack(level=1)  # date × stock
        pivot = pivot.reindex(index=self.dates, columns=self.stocks, fill_value=0.0)
        return pivot.values  # np.ndarray
    
    def _pivot_factors(self, factor_panel):
        """将因子面板转为 3D numpy 数组 (date × stock × factor)"""
        factor_names = factor_panel.columns.tolist()
        # 对每个因子分别 unstack，然后堆叠成 3D
        factor_3d = np.full((len(self.dates), len(self.stocks), len(factor_names)), np.nan)
        for f_idx, f_name in enumerate(factor_names):
            pivot = factor_panel[f_name].unstack(level=1)
            pivot = pivot.reindex(index=self.dates, columns=self.stocks)
            factor_3d[:, :, f_idx] = pivot.values
        return factor_3d
    
    def _get_daily_return_fast(self, date_idx, stock_idx_list):
        """O(1) 索引替代 O(N) 查找"""
        return self.returns_2d[date_idx, stock_idx_list]  # numpy 向量化
    
    def _get_factor_snapshot_fast(self, date_idx):
        """O(1) 行切片"""
        row = self.factors_2d[date_idx, :, :]  # (stock, factor)
        return pd.DataFrame(row, index=self.stocks, columns=self.factor_names)
    
    def _get_period_returns_fast(self, start_idx, end_idx, stock_idx_list):
        """用 log-return 求和累积"""
        # 直接用 numpy 切片 + 求和
        log_sum = self.log_returns_2d[start_idx+1:end_idx+1, stock_idx_list].sum(axis=0)
        return np.exp(log_sum) - 1
```

**收益**: 从每次 `xs()` 的 O(N) 降到 O(1)，直接 `numpy` 索引访问。这是最大的加速点。

**兼容性**: 保持现有 `.run()` 接口不变，内部替换为 fast 方法。

---

### 3.2 方案 B: 向量化回测循环 (L3)

**目标**: 不用逐日迭代，而是用矩阵运算一次计算所有调仓期的收益率。

**核心思路**:

对于月度调仓，每个调仓日 `t_i` 到 `t_{i+1}` 的持仓期，持仓是固定 `N` 只。整个回测可以看作：

```
权益 = 初始资本 × ∏_i [ (1 + ∑_{j=t_i}^{t_{i+1}} r_j × w) - 成本 ]
```

**向量化实现**:

```python
def run_vectorized(self, universe_mask, ranking_factor_idx, ascending):
    """
    universe_mask: (date_idx, stock_idx) → bool 的 2D numpy array
                   预计算: 每个调仓日哪些股票满足 universe 条件
    
    返回权益曲线和持仓记录
    """
    n_rebals = len(self.rebalance_dates)
    equity = 1.0
    equity_curve = np.zeros(len(self.dates))
    
    for i in range(n_rebals - 1):
        rebal_idx = self.date_to_idx[self.rebalance_dates[i]]
        next_idx = self.date_to_idx[self.rebalance_dates[i + 1]]
        
        # 1. 获取因子快照 (numpy 行)
        factor_vals = self.factors_2d[rebal_idx, :, ranking_factor_idx]
        
        # 2. 应用 universe 掩码
        valid_mask = universe_mask[i, :]  # 预计算
        factor_vals = np.where(valid_mask, factor_vals, np.nan)
        
        # 3. 排序选股 (numpy argsort)
        sorted_idx = np.argsort(factor_vals, kind='mergesort')  # 稳定排序
        if not ascending:
            sorted_idx = sorted_idx[::-1]
        picked_idx = sorted_idx[:self.n_stocks]  # 取前 N
        
        # 4. 计算持仓期累积收益率 (向量化)
        period_log_returns = self.log_returns_2d[rebal_idx+1:next_idx+1, picked_idx]
        cum_log = period_log_returns.sum(axis=0)
        stock_cum_returns = np.exp(cum_log) - 1
        portfolio_return = stock_cum_returns.mean()  # 等权
        
        # 5. 扣成本
        equity *= (1 + portfolio_return) * (1 - self.commission)
        
        # 6. 记录权益曲线
        equity_curve[rebal_idx:next_idx+1] = equity
    
    return equity_curve
```

**收益**: 将 inner loop 的 Python 逐日迭代改为 `numpy` 矩阵求和，C-level 执行。

---

### 3.3 方案 C: 预计算 Universe 和 中性化 (L2)

**目标**: 避免每次调仓都重新计算 `universe_filter` 和 `neutralize`。

**实现**:

```python
def precompute_universe(self, universe_filter_fn):
    """在 __init__ 或 run() 开始时一次性预计算所有调仓日的 universe"""
    self.universe_mask = np.zeros((len(self.rebalance_dates), len(self.stocks)), dtype=bool)
    for i, rebal_date in enumerate(self.rebalance_dates):
        snapshot = self._get_factor_snapshot_fast(self.date_to_idx[rebal_date])
        selected = universe_filter_fn(snapshot, self.dates, i)
        for s in selected:
            if s in self.stock_to_idx:
                self.universe_mask[i, self.stock_to_idx[s]] = True

def precompute_neutralized_factors(self, factor_cols, strength=0.5):
    """预计算所有调仓日的中性化因子，避免循环内重复 OLS"""
    for f_idx, f_name in enumerate(factor_cols):
        if f_name not in self.factor_names:
            continue
        col_idx = self.factor_names.index(f_name)
        
        for i, rebal_date in enumerate(self.rebalance_dates):
            rebal_idx = self.date_to_idx[rebal_date]
            snapshot = self._get_factor_snapshot_fast(rebal_idx)
            
            # 只对 universe 中的股票做中性化
            mask = self.universe_mask[i, :]
            factor = snapshot[f_name].values.copy()
            
            # 构建 controls (size + industry)
            controls = self._build_controls_fast(snapshot, mask)
            
            # OLS 回归取残差
            factor_neut = neutralize_ols_fast(factor, controls, mask)
            self.factors_2d[rebal_idx, :, col_idx] = factor_neut
```

**收益**: 中性化 OLS 从每次调仓 5-15ms 降到预计算后 0ms。

---

### 3.4 方案 D: 多策略并行 (L0)

**目标**: 多个策略共享同一批数据，但独立计算排名和回测，可以并行。

**实现**:

```python
from multiprocessing import Pool, cpu_count

def run_multiple_strategies(self, strategies):
    """
    strategies: [(name, universe_filter, ranking_factor, ascending), ...]
    
    共享数据，并行计算不同策略
    """
    # 预计算所有策略的 universe_mask（各不相同，可以串行）
    universe_masks = {}
    for name, uf, _, _ in strategies:
        if name not in universe_masks:
            universe_masks[name] = self._precompute_universe(uf)
    
    # 并行运行每个策略
    with Pool(min(cpu_count(), len(strategies))) as pool:
        args = [(self, name, universe_masks[name], rf, asc) 
                for name, _, rf, asc in strategies]
        results = pool.starmap(_run_single_strategy, args)
    
    return results
```

**收益**: 4核下 4 策略并行从串行 4x 时间降到 1x 时间。

---

### 3.5 方案 E: IC 和分位数计算向量化 (L3)

**目标**: `_compute_ic_series` 和 `_compute_quantile_spread` 当前全量循环，可以改为向量化。

**实现**:

```python
def _compute_ic_series_fast(self, ranking_factor_idx):
    """
    向量化 IC 计算：一次性对所有调仓日计算
    """
    ic_values = []
    ic_dates = []
    
    for i in range(len(self.rebalance_dates) - 1):
        rebal_idx = self.date_to_idx[self.rebalance_dates[i]]
        next_idx = self.date_to_idx[self.rebalance_dates[i + 1]]
        
        # 获取因子值和期间收益率（numpy 向量化）
        factor_vals = self.factors_2d[rebal_idx, :, ranking_factor_idx]
        period_log = self.log_returns_2d[rebal_idx+1:next_idx+1, :].sum(axis=0)
        period_ret = np.exp(period_log) - 1
        
        # 去掉 NaN，计算 Spearman rank correlation
        valid = ~np.isnan(factor_vals) & ~np.isnan(period_ret)
        if valid.sum() < 5:
            continue
        
        # numpy 实现的 rank correlation
        corr = spearman_rank_correlation(factor_vals[valid], period_ret[valid])
        if not np.isnan(corr):
            ic_values.append(corr)
            ic_dates.append(self.rebalance_dates[i])
    
    return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates))
```

---

## 4. Roadmap

### Phase 1: 数据层预计算 (高ROI, 1天)

| 任务 | 工时 | 优先级 | 说明 |
|-----|------|-------|------|
| 1.1 实现 `_pivot_returns()` 和 `_pivot_factors()` | 4h | P0 | 核心加速，所有优化依赖此步 |
| 1.2 实现 `date_to_idx` / `stock_to_idx` 映射 | 2h | P0 | 消除索引查找 overhead |
| 1.3 实现 `_get_factor_snapshot_fast()` 等 fast 方法 | 4h | P0 | 保持接口兼容 |
| 1.4 单元测试：确保 fast 方法输出与原方法一致 | 4h | P0 | 必须验证正确性 |

**里程碑**: 回测引擎内部全部使用 2D numpy 数组，预期 5,400 只股票回测从 10min → 2min。

### Phase 2: 回测循环向量化 (高ROI, 0.5天)

| 任务 | 工时 | 优先级 | 说明 |
|-----|------|-------|------|
| 2.1 用 `numpy` 矩阵运算替代 `.run()` 中的逐日循环 | 4h | P0 | 核心加速点 |
| 2.2 预计算 `log_returns_2d` 和 `cum_returns_2d` | 2h | P0 | 避免重复计算 exp/log |
| 2.3 用 `np.argsort` 替代 `pd.Series.nsmallest` | 2h | P1 | 更快速度 |
| 2.4 批量测试：多策略、多参数组合 | 4h | P1 | 确保正确性 |

**里程碑**: `run()` 内部不再逐日迭代，一次矩阵求和完成持仓期收益计算。预期再提速 2-3x。

### Phase 3: 预计算与缓存 (中ROI, 0.5天)

| 任务 | 工时 | 优先级 | 说明 |
|-----|------|-------|------|
| 3.1 预计算 `universe_mask` 矩阵 | 2h | P1 | 避免重复 universe 过滤 |
| 3.2 预计算中性化因子矩阵 | 4h | P1 | 避免循环内 OLS |
| 3.3 缓存 `returns_2d` / `factors_2d` 到磁盘 (.npy) | 2h | P2 | 下次加载秒级 |
| 3.4 缓存 `cum_returns_2d` | 1h | P2 | 避免重复计算 |

**里程碑**: 5,400 只股票数据加载 + 预计算从 10s → 3s（缓存后）。

### Phase 4: IC/分位数向量化 (中ROI, 0.5天)

| 任务 | 工时 | 优先级 | 说明 |
|-----|------|-------|------|
| 4.1 实现 `_compute_ic_series_fast()` | 2h | P1 | 向量化 rank correlation |
| 4.2 实现 `_compute_quantile_spread_fast()` | 2h | P1 | 向量化分位数切分 |
| 4.3 实现 `compute_ic_decay()` 向量化 | 2h | P2 | 可选 |
| 4.4 验证与旧方法结果一致 | 2h | P1 | 必须 |

### Phase 5: 多策略并行 (低ROI, 0.5天)

| 任务 | 工时 | 优先级 | 说明 |
|-----|------|-------|------|
| 5.1 实现 `run_multiple_strategies()` 并行 | 4h | P2 | 适合参数扫描 |
| 5.2 共享内存优化（避免 fork 时复制大数据） | 4h | P2 | 使用 `multiprocessing.shared_memory` |
| 5.3 支持 `concurrent.futures` 线程池替代 | 2h | P2 | 轻量并行 |

### Phase 6: 进阶优化 (可选, 0.5-1天)

| 任务 | 工时 | 优先级 | 说明 |
|-----|------|-------|------|
| 6.1 用 `numba` 加速核心循环 | 4h | P3 | 纯 numpy 不够快时启用 |
| 6.2 稀疏矩阵优化（停牌/退市股票零值） | 4h | P3 | 内存优化 |
| 6.3 增量更新机制（每天只增量计算新数据） | 4h | P3 | 日常回测场景 |
| 6.4 内存池管理（减少 GC 压力） | 2h | P3 | 长期运行稳定 |

---

## 5. 关键设计决策

### 5.1 保持向后兼容

所有优化在 `__init__` 内部完成，`.run()` 接口不变。添加 `use_fast=True` 开关：

```python
engine = CrossSectionalEngine(...)
# 新引擎自动使用 fast 方法
result = engine.run(...)  # 内部自动切换
```

### 5.2 数据一致性验证

每个 Phase 后必须验证：fast 输出与原始输出差异 < 1e-6（浮点误差范围）。

### 5.3 内存预算

5,400 只股票 × 1,200 日 × 8 bytes = 52MB/矩阵  
4 个 2D 矩阵 ≈ 200MB  
3D 因子矩阵 ≈ 500MB  
**总内存**: < 1GB，macOS 无压力。

### 5.4 缓存策略

```
data_cache/
  ├── *.parquet          # 原始个股数据
  ├── cache/
  │    ├── returns_2d.npy      # 日收益率矩阵
  │    ├── factors_2d.npy      # 因子矩阵
  │    ├── cum_returns_2d.npy  # 累积收益率
  │    └── meta.json           # 日期/股票列表 + hash
```

缓存 key: 所有 parquet 文件的 `mtime + size` 的 hash，文件变化时自动重新计算。

---

## 6. 风险与缓解

| 风险 | 等级 | 缓解 |
|-----|------|------|
| 向量化后结果与原始不同 | 高 | 每个 Phase 后都做 diff 测试 |
| 内存占用过高 | 低 | 200MB-1GB，现代机器无压力 |
| 缓存失效导致重复计算 | 中 | 用 mtime+size hash 做缓存 key |
| Numba 安装/兼容性 | 低 | 可选，Phase 6 才用 |
| 多进程 fork 复制大数据 | 中 | 用 shared_memory 或 spawn 模式 |

---

## 7. 验收标准

| 指标 | 当前 | 目标 | 测试方法 |
|-----|------|------|---------|
| 全量回测时间 | >10min | <2min | 5,400 只股票 × 4 策略 |
| 数据加载+预计算 | 10s | 3s | 含缓存 |
| 单策略回测 | 2.5min | <15s | Micro-cap 单策略 |
| 内存占用 | <2GB | <1.5GB | htop / ps |
| 输出一致性 | N/A | 差异<1e-6 | 与原始输出逐行比较 |
| 4 策略并行 | 4×串行时间 | 1×串行时间 | 4核并行 |

---

## 8. 下一步行动

按 Roadmap 执行，Phase 1 即可解决 80% 的性能问题。用户确认后，我可以立即开始 Phase 1 的编码实现。

---

*文档结束*
