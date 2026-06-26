"""P1-A 方案：修复因子 + 组合搜索 + 回测诊断

用法:
    cd thinking_and_learning_with_AI
    PYTHONPATH=. python tools/backtest_mvp/diagnose_p1_revised.py

功能:
    1. 加载数据并构建修复后的因子（轻量中性化 + A1/A3/A5修正）
    2. 用 Research Loop 搜索最优组合（权重/组合方式）
    3. 回测并输出诊断报告
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from itertools import combinations
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

from tools.backtest_mvp.engine import CrossSectionalEngine, BacktestResult
from tools.backtest_mvp.factors import load_price_data, compute_factors, load_daily_mcap_pb
from tools.backtest_mvp.data import DATA_DIR
from tools.backtest_mvp.p1_factor_mining import (
    NeutralizationPipeline, NewAlphaFactors, FactorZooEvaluator,
)
from tools.backtest_mvp.research_loop.deflated_sharpe import compute_deflated_sharpe

# ──────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────

def load_data():
    """加载已有数据"""
    data = load_price_data(str(DATA_DIR))
    mcap_pb = load_daily_mcap_pb(str(DATA_DIR))
    factor_panel, return_panel = compute_factors(data, mcap_pb_data=mcap_pb)
    return data, factor_panel, return_panel


# ──────────────────────────────────────────
# 组合搜索（Research Loop）
# ──────────────────────────────────────────

@dataclass
class ComboResult:
    """组合回测结果"""
    combo_name: str
    factor_weights: Dict[str, float]
    annual_return: float
    sharpe: float
    max_dd: float
    calmar: float
    dsr: float
    ic_ir: float
    
    def to_dict(self):
        return asdict(self)


class FactorComboSearch:
    """Research Loop: 搜索最优因子组合"""
    
    def __init__(self, factor_panel: pd.DataFrame, return_panel: pd.DataFrame):
        self.factor_panel = factor_panel
        self.return_panel = return_panel
        self.candidate_factors = {}
        self.candidate_stats = pd.DataFrame()
    
    def build_all_factors(self) -> Dict[str, pd.Series]:
        """构建所有候选因子"""
        factors = {}
        
        # 已有因子（数值列）
        existing_cols = [c for c in self.factor_panel.columns 
                        if c not in ["close", "open", "high", "low", "volume", "symbol", "date", "industry"]]
        for col in existing_cols:
            if pd.api.types.is_numeric_dtype(self.factor_panel[col]):
                factors[f"existing_{col}"] = self.factor_panel[col]
        
        # 新 Alpha 因子
        print("  [P1] 构建新因子...")
        pipeline = NeutralizationPipeline(self.factor_panel)
        new_alpha = NewAlphaFactors(self.factor_panel, pipeline)
        new_factors = new_alpha.get_all_factors()
        factors.update(new_factors)
        
        # 评估所有因子
        print("  [P1] 评估因子...")
        evaluator = FactorZooEvaluator(self.factor_panel, self.return_panel)
        self.candidate_stats = evaluator.evaluate_all(factors, min_ic_ir=0.15)
        
        self.candidate_factors = factors
        return factors
    
    def search_combinations(
        self, max_factors: int = 3, min_ic_ir: float = 0.15, n_trials: int = 50,
    ) -> List[ComboResult]:
        """搜索最优组合"""
        valid_factors = self.candidate_stats[
            self.candidate_stats["ic_ir"] > min_ic_ir
        ]["factor"].tolist()
        
        print(f"\n  有效因子: {len(valid_factors)}/{len(self.candidate_factors)}")
        print(f"  {valid_factors}")
        
        if len(valid_factors) < 2:
            print("  ⚠️ 有效因子不足2个，无法搜索组合")
            return []
        
        results = []
        trial_count = 0
        
        for n in range(2, min(max_factors + 1, len(valid_factors) + 1)):
            for combo in combinations(valid_factors, n):
                if trial_count >= n_trials:
                    break
                
                weight_schemes = self._generate_weights(combo)
                
                for weights in weight_schemes:
                    trial_count += 1
                    if trial_count > n_trials:
                        break
                    
                    combined = self._combine_factors(combo, weights)
                    result = self._backtest_combo(combined, combo, weights)
                    if result:
                        results.append(result)
                
                if trial_count >= n_trials:
                    break
        
        results.sort(key=lambda x: x.sharpe, reverse=True)
        return results
    
    def _generate_weights(self, factors: Tuple[str, ...]) -> List[Dict[str, float]]:
        """生成权重方案"""
        n = len(factors)
        schemes = []
        
        # 等权
        schemes.append({f: 1.0 / n for f in factors})
        
        # IC 加权
        ic_ir_vals = []
        for f in factors:
            row = self.candidate_stats[self.candidate_stats["factor"] == f]
            if len(row) > 0:
                ic_ir_vals.append(max(row["ic_ir"].values[0], 0.01))
            else:
                ic_ir_vals.append(0.01)
        
        total = sum(ic_ir_vals)
        schemes.append({f: ic_ir_vals[i] / total for i, f in enumerate(factors)})
        
        return schemes
    
    def _combine_factors(self, factors: Tuple[str, ...], weights: Dict[str, float]) -> pd.Series:
        """合成因子"""
        combined = pd.Series(0.0, index=self.factor_panel.index)
        for f in factors:
            if f in self.candidate_factors:
                w = weights.get(f, 0.0)
                combined += self.candidate_factors[f].fillna(0) * w
        return combined
    
    def _backtest_combo(
        self, combined_factor: pd.Series, combo: Tuple[str, ...], weights: Dict[str, float]
    ) -> Optional[ComboResult]:
        """回测组合"""
        # 将合成因子加入面板
        self.factor_panel["combined_signal"] = combined_factor
        
        engine = CrossSectionalEngine(
            factor_panel=self.factor_panel,
            return_panel=self.return_panel,
            initial_capital=1.0,
            n_stocks=30,
            rebalance_freq='M',
            commission=0.00125,
            slippage=0.002,
        )
        
        try:
            result = engine.run(
                ranking_factor="combined_signal",
                ascending=False,
            )
            
            n_months = len(result.monthly_returns) if hasattr(result, 'monthly_returns') and result.monthly_returns is not None else 120
            dsr = compute_deflated_sharpe(sharpe=result.sharpe_ratio, n_trials=50, n_periods=max(n_months, 2))
            calmar = result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0
            
            return ComboResult(
                combo_name="+".join(combo),
                factor_weights=weights,
                annual_return=result.annual_return,
                sharpe=result.sharpe_ratio,
                max_dd=result.max_drawdown,
                calmar=calmar,
                dsr=dsr,
                ic_ir=0.0,
            )
        except Exception as e:
            print(f"    回测失败: {e}")
            return None


# ──────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────

def main():
    print("=" * 70)
    print("  P1-A 方案：修复因子 + 组合搜索")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    data, factor_panel, return_panel = load_data()
    print(f"  面板: {factor_panel.shape[0]} 行, {factor_panel.shape[1]} 列")
    
    # 2. 构建因子
    print("\n[2/4] 构建因子（修复版）...")
    searcher = FactorComboSearch(factor_panel, return_panel)
    all_factors = searcher.build_all_factors()
    print(f"  总因子: {len(all_factors)}")
    
    # 查看修复效果
    print("\n  修复后因子评估:")
    stats = searcher.candidate_stats
    for _, row in stats.iterrows():
        if row["factor"].startswith("A"):
            print(f"    {row['factor']:<35} IC={row['ic_mean']:>+.3f} IR={row['ic_ir']:>.3f} "
                  f"t={row['ic_tstat']:>6.2f} Sharpe={row['sharpe']:>.2f} "
                  f"{'✓' if row['valid'] else '✗'}")
    
    # 3. 组合搜索
    print("\n[3/4] 组合搜索（Research Loop）...")
    results = searcher.search_combinations(max_factors=3, min_ic_ir=0.15, n_trials=50)
    
    # 4. 输出结果
    print("\n[4/4] 诊断报告")
    print("=" * 70)
    
    if not results:
        print("  ⚠️ 未找到有效组合")
        return
    
    # 筛选可部署组合
    deployable = [r for r in results if r.sharpe >= 1.0 and r.max_dd > -0.30 and r.dsr > 0]
    
    print(f"\n  搜索完成: {len(results)} 个组合")
    print(f"  可部署组合 (Sharpe>=1.0, 回撤<-30%, DSR>0): {len(deployable)}")
    
    if deployable:
        print("\n  可部署组合 Top 3:")
        print("  " + "-" * 66)
        for i, r in enumerate(deployable[:3], 1):
            print(f"  #{i} {r.combo_name}")
            print(f"     权重: { {k: f'{v:.2f}' for k, v in r.factor_weights.items()} }")
            print(f"     年化: {r.annual_return:.2%} | Sharpe: {r.sharpe:.2f} | 回撤: {r.max_dd:.2%}")
            print(f"     Calmar: {r.calmar:.2f} | DSR: {r.dsr:.2f}")
            print()
    
    # 全部 Top 5
    print("\n  全部 Top 5（按 Sharpe）:")
    print("  " + "-" * 66)
    print(f"  {'排名':<4} {'组合':<30} {'年化':<8} {'Sharpe':<8} {'回撤':<8} {'DSR':<6}")
    print("  " + "-" * 66)
    for i, r in enumerate(results[:5], 1):
        deploy_mark = "✓" if r in deployable else ""
        print(f"  {i:<4} {r.combo_name:<30} {r.annual_return:<8.2%} {r.sharpe:<8.2f} {r.max_dd:<8.2%} {r.dsr:<6.2f} {deploy_mark}")
    
    # 保存 CSV
    output_csv = Path(__file__).parent / "diagnose_p1_revised_report.csv"
    pd.DataFrame([r.to_dict() for r in results]).to_csv(output_csv, index=False)
    print(f"\n  详细报告: {output_csv}")
    
    print("\n" + "=" * 70)
    print("  P1-A 诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
