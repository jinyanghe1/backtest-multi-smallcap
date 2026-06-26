# ML 因子挖掘与动态策略调研报告

**调研日期**: 2026-06-25  
**背景**: 小资金约束（日频数据、≤1000只微盘股、有限计算资源、无GPU）

---

## 核心问题

如何在**不断变化的市场环境**中保持策略有效性？

股票 ≠ 图像识别：
- 图像识别的分布相对稳定（猫的图始终像猫）
- 股票市场的分布持续漂移（2021年的小盘因子 ≠ 2024年的小盘因子）
- 所以：静态模型 → 快速失效；动态模型 → 过拟合

---

## 1. 四大动态策略方法

### 1.1 Rolling Window Training（滚动窗口训练）

| 项目 | 详情 |
|:--|:--|
| **原理** | 只用最近N天的数据训练模型，每隔M天重新训练 |
| **窗口大小** | 通常252日（1年）到756日（3年） |
| **重训练频率** | 月度或季度 |
| **数据需求** | 日频（已有） |
| **计算量** | **低**（每次只训练小模型） |
| **可落地性** | ⭐⭐⭐⭐⭐ 极高 |

**为什么这是最适合小资金的方案**：
- 不需要额外数据
- 不需要GPU（CPU即可）
- 逻辑简单：丢旧数据、用新数据、重新训练
- 天然符合 walk-forward 验证框架

**A股微盘适用性**：
- 微盘股的alpha衰减周期约6-12个月
- 所以窗口不宜太长（1-2年最优）
- 太短（3个月）则噪声过大

**具体实现**：
```python
# 每季度初重新训练
train_window = 252  # 1年数据
retrain_freq = 63   # 每季度

for date in rebalance_dates:
    if date - last_retrain_date >= retrain_freq:
        train_data = data[(date - train_window) : date]
        model.fit(train_data)  # 重新训练
        last_retrain_date = date
    
    predictions = model.predict(date_data)
```

---

### 1.2 Online Learning（在线学习）

| 项目 | 详情 |
|:--|:--|
| **原理** | 每来一个新数据点，只更新模型参数，不重新训练整个模型 |
| **代表算法** | SGD（随机梯度下降）、Bayesian Ridge、Online Random Forest |
| **数据需求** | 日频（已有） |
| **计算量** | **极低**（每次只更新参数） |
| **可落地性** | ⭐⭐⭐⭐⭐ 极高 |

**为什么比 Rolling Window 更适合实盘**：
- Rolling Window 每季度需要"全量重训练"（虽然数据少，但还是要重新fit）
- Online Learning 每天只需要"增量更新"（一个梯度步）
- 对实时性要求高的策略更友好

**推荐的在线学习算法**：

| 算法 | 优点 | 缺点 | 适用场景 |
|:--|:--|:--|:--|
| **SGD Regressor** | 极快、内存占用小 | 需要调学习率 | 线性因子模型 |
| **Passive-Aggressive** | 自动调学习率 | 对噪声敏感 | 高信噪比环境 |
| **Online Random Forest** | 非线性、鲁棒 | 需要维护多棵树 | 复杂因子交互 |
| **Bayesian Ridge** | 自动正则化、不确定性估计 | 较慢 | 小样本（微盘股） |

**小资金推荐**：**SGD Regressor + 手动衰减学习率**

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.001,  # 正则化强度
    learning_rate='adaptive',
    eta0=0.01,
)

# 每天增量更新
for day in data:
    X = get_factors(day)  # 当日因子
    y = get_next_day_return(day)  # 次日收益（标签）
    model.partial_fit(X, y)  # 在线学习！
```

---

### 1.3 Regime-Switching Models（状态切换模型）

| 项目 | 详情 |
|:--|:--|
| **原理** | 识别市场处于"高波动/低波动"、"牛市/熊市"，在不同状态下使用不同策略 |
| **代表算法** | HMM（隐马尔可夫）、SJM（Sparse Jump Model）、K-Means Regime |
| **数据需求** | 日频 + 波动率/成交量等状态指标 |
| **计算量** | 中等（HMM需要EM算法迭代） |
| **可落地性** | ⭐⭐⭐⭐ 高 |

**为什么这对A股微盘特别重要**：
- 微盘股在牛市中表现极好（小盘效应放大）
- 微盘股在熊市中暴跌（流动性危机）
- 如果能识别"熊市来临"，可以空仓或减仓，避免-40%回撤

**已有实现**：
- 你的 `backtest_mvp` 已有 `regime_momentum` 模板（SJM-based）
- 但效果不佳（可能因窗口太短或参数不匹配）

**简化版（小资金可落地）**：

```python
def detect_regime(volatility_series, window=60):
    """
    简单状态识别：基于历史波动率分位数
    """
    current_vol = volatility_series[-window:].mean()
    historical_vol = volatility_series.mean()
    vol_ratio = current_vol / historical_vol
    
    if vol_ratio > 1.5:
        return "high_vol"  # 高波动 → 用反转策略
    elif vol_ratio < 0.7:
        return "low_vol"   # 低波动 → 用动量策略
    else:
        return "normal"    # 正常 → 用默认策略

# 策略切换表
regime_strategy_map = {
    "high_vol": strategy_reversal,    # 高波动做反转（抄底）
    "low_vol": strategy_momentum,     # 低波动做动量（趋势）
    "normal": strategy_default,       # 正常用默认（小市值）
}
```

---

### 1.4 Meta-Learning / 学习动态权重（元学习）

| 项目 | 详情 |
|:--|:--|
| **原理** | 不预测股票收益，而是预测"哪个因子在当前市场有效" |
| **代表算法** | 因子动量（Factor Momentum）、动态因子权重 |
| **数据需求** | 日频 + 因子历史IC序列 |
| **计算量** | 中等（需要维护因子IC数据库） |
| **可落地性** | ⭐⭐⭐⭐ 高 |

**核心洞察**：
- 你的回测结果显示：在微盘里，市值因子始终有效，但动量因子始终无效
- 如果能动态调整因子权重（市值因子权重高、动量因子权重低），可以提升稳健性

**已有实现**：
- 你的 S11（因子动量动态权重）尝试了这个，但效果差
- 原因：因子动量本身在微盘里不存在，所以"预测哪个因子有效"也失效了
- 改进方向：用**短期IC**（过去3个月）而非**长期IC**（过去1年）来动态调整权重

---

## 2. 小资金可落地的 ML 因子挖掘方案

### 推荐方案：SGD Online Learning + Regime Filter

**架构**：
```
数据层
  ↓
特征层：市值、PB、PE、动量、波动率、换手率、Peer Momentum（新增）
  ↓
Regime Detector：基于60日波动率识别市场状态
  ↓
ML 模型层：SGD Regressor（在线学习）
  ↓
信号层：预测收益 → 排序 → 选Top 30
  ↓
执行层：月度调仓 + 日度止损
```

**为什么这是最优方案**：

| 约束 | 方案如何满足 |
|:--|:--|
| 存储限制 | 只保留1年数据（252日×586只×10因子 ≈ 1.5MB） |
| 数据获取限制 | 只用日频数据（已有） |
| 计算能力限制 | SGD在线学习每步 < 1ms（CPU即可） |
| 过拟合风险 | L2正则化 + 滚动验证 + 早停 |

---

## 3. 训练-测试划分的科学方法

### 核心原则：时序不可穿越

**错误做法**：
```
随机打乱所有数据 → 训练/测试划分
问题：2026年的数据"泄露"到2023年的训练中
```

**正确做法**：
```
Train: 2018-2023  →  验证/调参
Val: 2023-2024   →  选择超参数（窗口大小、正则化强度）
Test: 2024-2026  →  最终验证（只能用一次！）
```

### 你的 walk_forward 框架已正确实现

```python
# 已实现（walk_forward.py）
split_by_date(
    factor_panel, return_panel,
    train_end="2024-01-01",  # 训练截止
    val_end=None,             # 可选验证期
)
```

**关键规则**：
1. 超参数选择（如正则化强度、窗口大小）只能在 **Val 期** 上验证
2. **Test 期只能跑一次**——如果跑多次并选择最优结果，就是间接的数据泄露
3. 如果 Test 结果不好，**不能回去调整参数再重新跑 Test**——必须接受结果

### 推荐的三阶段划分

| 阶段 | 时间 | 用途 | 数据量 | 可访问次数 |
|:--|:--|:--|:--|:--|
| Train | 2018-2023.12 | 模型训练、因子挖掘 | 5年 | 无限 |
| Val | 2024.01-2024.12 | 超参数选择、模型选择 | 1年 | 有限（<10次）|
| Test | 2025.01-2026.06 | 最终验证（不可用于调参）| 1.5年 | **只能1次** |

---

## 4. 防止过拟合的5层防线

### 防线1：简单模型（Occam's Razor）

**不要用的**：
- ❌ 深度神经网络（DNN、LSTM）
- ❌ 复杂的GBDT（XGBoost、LightGBM，树深>5）
- ❌ 高维因子组合（因子数 > 20）

**推荐用的**：
- ✅ 线性模型（SGD、Ridge）
- ✅ 浅层树（max_depth=3）
- ✅ 因子数 ≤ 10

> 在微盘股（586只）中，样本量有限，复杂模型必然过拟合

### 防线2：强正则化

```python
from sklearn.linear_model import Ridge

model = Ridge(
    alpha=1.0,  # 强正则化
    fit_intercept=False,  # 零均值收益假设
)
```

### 防线3：Purged K-Fold（清洗交叉验证）

传统K-Fold：随机划分 → 时序泄露  
Purged K-Fold：在训练集和测试集之间留出"缓冲期"（gap），防止信息泄露

```python
def purged_kfold_split(dates, n_splits=5, purge_gap=10):
    """
    时序交叉验证：训练集和测试集之间留出 purge_gap 天
    """
    fold_size = len(dates) // n_splits
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        # 训练集 = 测试集之前，但留出 purge_gap
        train_end = test_start - purge_gap
        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]
        
        yield train_dates, test_dates
```

### 防线4：因子正交化（去除共线性）

如果两个因子高度相关（如市值和PE在微盘股中常常负相关），模型会不稳定。

```python
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr

def orthogonalize_factors(factors_df):
    """对因子进行正交化（Gram-Schmidt）"""
    X = StandardScaler().fit_transform(factors_df)
    Q, R = qr(X, mode='economic')
    return pd.DataFrame(Q, index=factors_df.index, columns=factors_df.columns)
```

### 防线5：样本外稳定性测试

```python
def test_stability(result, min_months=6):
    """
    测试策略在不同子期的稳定性
    """
    monthly = result.monthly_returns
    if len(monthly) < min_months * 2:
        return False
    
    mid = len(monthly) // 2
    first_half = monthly[:mid]
    second_half = monthly[mid:]
    
    # 如果前后两半的表现差异太大，说明不稳定
    if abs(first_half.mean() - second_half.mean()) > 0.02:
        return False
    
    return True
```

---

## 5. 具体实施路线图

### 阶段1：搭建ML框架（1周）
- [ ] 在 `backtest_mvp` 中新增 `ml_models.py` 模块
- [ ] 实现 SGD Online Regressor 封装
- [ ] 实现 Regime Detector（基于波动率）
- [ ] 实现特征工程（市值、PB、动量、波动率、换手率、Peer Momentum）

### 阶段2：训练与验证（2周）
- [ ] 在 Train 期（2018-2023）上训练模型
- [ ] 在 Val 期（2024）上选择超参数（正则化强度、窗口大小）
- [ ] 跑 Test 期（2025-2026）验证一次
- [ ] 对比：ML模型 vs 纯市值因子（S3）

### 阶段3：整合到实盘框架（1周）
- [ ] 将ML模型集成到 `run.py` 的 walk-forward 流程中
- [ ] 实现每日自动更新模型（在线学习）
- [ ] 实现 regime filter 自动切换策略

---

## 6. 预期效果与风险

| 指标 | 预期 |
|:--|:--|
| 年化收益 | 相比纯市值因子（S3: 41.9%），可能提升 3-8% |
| 夏普比率 | 可能提升 0.1-0.3（因为ML可以优化权重组合） |
| 最大回撤 | 通过Regime Filter可能降低 5-10% |
| 计算成本 | 每天 < 1秒（CPU） |
| 存储成本 | 每年 < 10MB |

| 风险 | 说明 |
|:--|:--|
| ML模型过拟合 | 即使做了正则化，小样本+高噪声仍然可能过拟合 |
| Regime识别滞后 | 波动率上升通常出现在下跌之后，可能"慢半拍" |
| 特征漂移 | 微盘股的因子有效性在2020年前后发生了结构性变化 |
| 数据质量 | 日频数据可能存在幸存者偏差、前视偏差 |

---

## 7. 总结

| 结论 | 说明 |
|:--|:--|
| **最推荐** | SGD Online Learning + Regime Filter |
| **数据需求** | 日频（已有） |
| **计算量** | 极低（CPU每天<1秒） |
| **存储需求** | 极小（<10MB/年） |
| **预期增益** | +3-8% 年化，夏普+0.1-0.3 |
| **优先实施** | 先在S3（纯市值）基础上增加ML层，验证增量价值 |
| **不推荐的** | 深度神经网络、GBDT、GNN（计算量过大、过拟合风险） |
| **关键风险** | 过拟合 → 需要强正则化 + 严格的Walk-forward验证 |
