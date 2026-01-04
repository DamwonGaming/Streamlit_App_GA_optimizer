#ga_optimizer
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from strategy import double_ma_strategy, calculate_metrics

class MAProblem(ElementwiseProblem):
    def __init__(self, df):
        self.df = df
        self.eval_count = 0
        self.effective_start_date = df.index[200]
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=1,
            xl=[5.0, 20.0],
            xu=[50.0, 200.0],
            vtype=float
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_count += 1
        
        short = int(np.clip(np.round(x[0]), 5, 50))
        long_ = int(np.clip(np.round(x[1]), 20, 200))
        
        if long_ - short < 20:
            long_ = short + 20
            if long_ > 200:
                long_ = 200
                short = long_ - 20
                if short < 5:
                    short = 5
        
        g = short - long_ + 20
        
        try:
            df_strat = double_ma_strategy(self.df, short, long_, drop_na=True)
            metrics = calculate_metrics(df_strat, start_date=self.effective_start_date)
            
            sharpe = metrics.get('sharpe_ratio', 0.0)
            if sharpe is None or np.isnan(sharpe):
                sharpe = 0.0
            
            mdd = abs(metrics.get('max_drawdown', 1.0))
            if mdd is None or np.isnan(mdd):
                mdd = 1.0
            
            out["F"] = [-sharpe, mdd]
            out["G"] = [g]
            
        except Exception as e:
            if self.eval_count % 100 == 0:
                print(f"第{self.eval_count}次评估失败: {e}")
            out["F"] = [0.0, 1.0]
            out["G"] = [1.0]

def run_ga_optimizer(df, n_gen=100, pop_size=100):
    problem = MAProblem(df)
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=LatinHypercubeSampling(),
        crossover=SBX(prob=0.8, eta=20, vtype=float),
        mutation=PM(prob=0.3, eta=30, vtype=float),
        repair=RoundingRepair(),
        eliminate_duplicates=True
    )
    
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=42,
        verbose=False,
        save_history=False
    )
    
    valid_idx = res.F[:, 0] < 0
    if np.any(valid_idx):
        F_valid = res.F[valid_idx]
        X_valid = res.X[valid_idx]
    else:
        F_valid = res.F
        X_valid = res.X
    
    if len(F_valid) > 0:
        F_norm = F_valid.copy()
        F_norm[:, 0] = (F_norm[:, 0] - F_norm[:, 0].min()) / (F_norm[:, 0].max() - F_norm[:, 0].min() + 1e-10)
        F_norm[:, 1] = (F_norm[:, 1] - F_norm[:, 1].min()) / (F_norm[:, 1].max() - F_norm[:, 1].min() + 1e-10)
        
        ideal = np.array([0, 0])
        distances = np.sqrt(np.sum((F_norm - ideal)**2, axis=1))
        best_idx = np.argmin(distances)
        
        best_sharpe_idx = np.argmax(-F_valid[:, 0])
        best_mdd_idx = np.argmin(F_valid[:, 1])
        
        print(f"\n=== 优化结果 ===")
        print(f"评估次数: {problem.eval_count}")
        print(f"帕累托解: {len(F_valid)}个")
        
        return F_valid, X_valid, {
            'compromise_idx': best_idx,
            'best_sharpe_idx': best_sharpe_idx,
            'best_mdd_idx': best_mdd_idx
        }
    
    return res.F, res.X, None
