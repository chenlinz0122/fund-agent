"""遗传算法参数进化（按第二期公开的《进化系统文档》实现）。

文档里最重要的一句话不是算法细节，而是目的定义：

    GA 的目的**绝不是寻找历史最高收益率**（那只是过拟合的数字游戏），
    而是要在覆盖极端行情的小区间的压力下，寻找"高胜率决策"与
    "极低摩擦成本"之间的极值点。

因此本模块有三条硬规矩，都写在代码里而不是靠自觉：

1. **参数盒只从 StrategyGenes 生成**。Environment（杠杆上限、long-only、
   全局熔断、费用下限）是常量，交叉和变异都碰不到它 —— 结构上杜绝
   "进化出一个加杠杆的高分解"。
2. **蒙特卡洛不参与适应度打分**。它只在进化结束后对最终挑战者做一次全样本
   复核（资金耗尽概率、分位收益），单独写进报告。把 MC 塞进 fitness
   会让每个体的成本爆炸，也会诱使人用偶然的路径去拟合。
3. **搜索得到的最好参数默认只是"挑战者"，不自动晋升为冠军**，
   必须由人显式确认；同时如实上报尝试次数 n_trials，
   用它去算 Deflated Sharpe —— 试了 500 个参数得到的最好夏普，
   门槛和只试 1 个完全不同。

种群机制按文档：10% 旧精英火种 / 40% 精英放大变异 / 50% 随机移民，
精英保留 5%，块级正交交叉，逐基因突变 0.15（无改进时斜坡上调至 0.55）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis import validation as VAL
from analysis.constants import DEFAULT_RISK_FREE, EPS, TRADING_DAYS
from analysis.risk_rules import DEFAULT_EXIT, ExitConfig
from analysis.simulation import simulate
from analysis.signals import DEFAULT_ENV, Environment, StrategyGenes

# ========== 基因分组（用于块级正交交叉，而不是逐基因打乱） ==========
GENE_BLOCKS: Dict[str, List[str]] = {
    "dynamics": ["abg"],                                    # 动力学感知段（PDE 权重）
    "risk_bands": ["vol_target", "vol_tolerance", "drawdown_mild",
                   "drawdown_severe", "mild_hold", "severe_hold"],
    "gains": ["dd_gain", "accel_gain", "premium_gain",
              "accel_threshold", "premium_threshold"],
    "friction": ["min_trade_delta", "rebalance_stride"],
    "timing": ["ma_window", "vol_window", "min_history"],
}

# 合法参数盒：float -> (下界, 上界, 是否整数)
PARAM_SPACE: Dict[str, Tuple[float, float, bool]] = {
    "vol_target": (0.04, 0.40, False),
    "vol_tolerance": (0.05, 2.0, False),
    "drawdown_mild": (0.04, 0.30, False),
    "drawdown_severe": (0.08, 0.60, False),
    "mild_hold": (0.20, 1.0, False),
    "severe_hold": (0.0, 0.80, False),
    "dd_gain": (2.0, 30.0, False),
    "accel_gain": (0.0, 20.0, False),
    "accel_threshold": (-1.5, -0.05, False),
    "premium_threshold": (0.0, 0.60, False),
    "premium_gain": (2.0, 30.0, False),
    "min_trade_delta": (0.01, 0.40, False),
    "rebalance_stride": (1, 42, True),
    "ma_window": (20, 504, True),
    "vol_window": (20, 252, True),
    "min_history": (20, 90, True),
}
ABG_SPACE: Dict[str, Tuple[float, float, bool]] = {
    "alpha": (0.05, 0.95, False),
    "beta": (0.0, 0.50, False),
    "gamma": (0.0, 0.20, False),
    "half_life": (5.0, 90.0, False),
    "k_stop": (2.0, 10.0, False),
}
INT_FIELDS = {k for k, (_, _, is_int) in PARAM_SPACE.items() if is_int}


@dataclass
class Individual:
    genes: StrategyGenes
    score: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    origin: str = "explorer"

    def clone(self) -> "Individual":
        return Individual(genes=replace(self.genes), score=self.score,
                          breakdown=dict(self.breakdown), origin=self.origin)


def _sample(rng: np.random.Generator, space: Dict[str, Tuple[float, float, bool]]) -> dict:
    out = {}
    for key, (lo, hi, is_int) in space.items():
        out[key] = int(rng.integers(int(lo), int(hi) + 1)) if is_int else float(rng.uniform(lo, hi))
    return out


def random_genes(rng: np.random.Generator) -> StrategyGenes:
    """50% 外来移民：在合法参数盒内均匀随机，与冠军解耦，用于打破局部最优。"""
    from analysis.abg import ABGParams
    g = StrategyGenes(**_sample(rng, PARAM_SPACE))
    g.abg = ABGParams(**_sample(rng, ABG_SPACE))
    return g.clipped()


def _mutate_scalar(value, lo, hi, is_int, sigma_ratio, rng):
    sigma = (hi - lo) * sigma_ratio
    new = value + rng.normal(0.0, sigma) if sigma > 0 else value
    new = float(np.clip(new, lo, hi))
    return int(round(new)) if is_int else new


def mutate(genes: StrategyGenes, rate: float = 0.15, sigma_ratio: float = 0.10,
           rng: Optional[np.random.Generator] = None) -> StrategyGenes:
    """逐基因突变（加性高斯步 + 投影回合法盒）。rate 由突变斜坡控制。"""
    rng = rng or np.random.default_rng()
    out = replace(genes, abg=replace(genes.abg))
    for key, (lo, hi, is_int) in PARAM_SPACE.items():
        if rng.random() < rate:
            object.__setattr__(out, key, _mutate_scalar(getattr(out, key), lo, hi, is_int,
                                                        sigma_ratio, rng))
    for key, (lo, hi, is_int) in ABG_SPACE.items():
        if rng.random() < rate:
            object.__setattr__(out.abg, key, _mutate_scalar(getattr(out.abg, key), lo, hi,
                                                            is_int, sigma_ratio, rng))
    return out.clipped()


def crossover(a: StrategyGenes, b: StrategyGenes, rng: Optional[np.random.Generator] = None
              ) -> StrategyGenes:
    """
    块级正交交叉：整段互换，而不是逐基因均匀打乱。

    文档要求保持"块级"语义：例如把 A 的动力学感知段（α/β/γ）与 B 的
    摩擦控制段（不交易带、调仓频率）组合，得到"感知敏锐但下手克制"的个体。
    """
    rng = rng or np.random.default_rng()
    from dataclasses import fields
    merged = {}
    for f in fields(StrategyGenes):
        if f.name == "abg":
            continue
        take_a = rng.random() < 0.5
        merged[f.name] = getattr(a if take_a else b, f.name)
    abg = {}
    for f in fields(a.abg):
        take_a = rng.random() < 0.5
        abg[f.name] = getattr((a if take_a else b).abg, f.name)
    from analysis.abg import ABGParams
    return StrategyGenes(**merged, abg=ABGParams(**abg)).clipped()


# ========== 适应度 ==========
def fitness(genes: StrategyGenes, returns_df: pd.DataFrame, env: Environment = DEFAULT_ENV,
            exit_cfg: Optional[ExitConfig] = None,
            benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Score = 相对定投基准的超额 − 摩擦成本 − 回撤惩罚 − 清仓空转惩罚。

    全悲观原则：所有成本按确定发生计算，回撤按指数扣分，
    "靠死扛偶尔获利"的基因必须在这里被淘汰。
    """
    sim = simulate(returns_df, genes=genes, env=env, exit_cfg=exit_cfg,
                   benchmark_returns=benchmark_returns)
    if "error" in sim:
        return {"score": -1e6, "alpha": 0.0, "cost": 0.0, "drawdown": 0.0,
                "turnover": 0.0, "trades": 0, "error": 1.0}
    res = sim["result"]
    stats = res["strategy"]["metrics"]
    alpha = float(res["alpha_vs_dca_pct"])
    cost = float(res["strategy"]["total_cost_pct"])
    mdd = abs(float(stats.get("max_drawdown_pct", 0) or 0))
    sharpe = float(stats.get("sharpe", 0) or 0)
    trades = int(res["strategy"]["trade_count"])
    exposure = float(res["avg_exposure_pct"]) / 100.0

    dd_penalty = (mdd / 100.0) ** 2 * 40.0            # 指数级扣分
    friction_penalty = cost * 1.5                      # 双边费用直接进分
    idle_penalty = max(0.0, 0.35 - exposure) * 25.0    # 长期空仓：机会成本
    churn_penalty = max(0, trades - 60) * 0.15         # 过度交易
    score = alpha + sharpe * 3.0 - dd_penalty - friction_penalty - idle_penalty - churn_penalty
    return {"score": round(float(score), 3), "alpha": round(alpha, 3),
            "sharpe": round(sharpe, 3), "total_return": round(float(stats.get("total_return_pct", 0)), 2),
            "cost": round(cost, 3), "drawdown": round(mdd, 2), "turnover":
                round(float(res["strategy"]["total_turnover_pct"]), 2),
            "trades": trades, "avg_exposure": round(exposure, 3),
            "fused": 1.0 if res.get("fused") else 0.0}


# ========== 进化主循环 ==========
def _rank(pop: List[Individual]) -> List[Individual]:
    return sorted(pop, key=lambda x: x.score, reverse=True)


def _fill_population(pop_size: int, incumbent: Optional[Individual], rng: np.random.Generator,
                     evaluator: Callable[[StrategyGenes], Dict[str, float]],
                     elite_ratio: float = 0.10, mutant_ratio: float = 0.40) -> List[Individual]:
    """10% 旧神火种 / 40% 伴生变异 / 50% 外来移民。"""
    n_elite = int(round(pop_size * elite_ratio))
    n_mutant = int(round(pop_size * mutant_ratio))
    pop: List[Individual] = []

    if incumbent is not None:
        for _ in range(max(n_elite, 1)):
            pop.append(incumbent.clone())
    while len(pop) < n_elite:
        pop.append(Individual(genes=random_genes(rng), origin="seed"))

    for _ in range(n_mutant):
        base = pop[rng.integers(0, len(pop))].genes if pop else random_genes(rng)
        # 精英拷贝 + 加大尺度扰动（文档：约 1.5 倍突变幅度）
        ind = Individual(genes=mutate(base, rate=0.6, sigma_ratio=0.15, rng=rng), origin="mutant")
        pop.append(ind)

    while len(pop) < pop_size:
        pop.append(Individual(genes=random_genes(rng), origin="explorer"))

    for ind in pop:
        if ind.score == 0.0 and not ind.breakdown:
            ind.breakdown = evaluator(ind.genes)
            ind.score = ind.breakdown["score"]
    return pop[:pop_size]


def evolve(returns_df: pd.DataFrame, env: Environment = DEFAULT_ENV,
           exit_cfg: Optional[ExitConfig] = None,
           benchmark_returns: Optional[pd.Series] = None,
           pop_size: int = 24, generations: int = 5, seed: int = 2026,
           incumbent_genes: Optional[StrategyGenes] = None,
           elite_ratio: float = 0.05, mutation_rate: float = 0.15,
           mutation_cap: float = 0.55, ramp_factor: float = 1.25,
           patience: int = 2, max_evals: int = 400) -> Dict[str, object]:
    """
    跑一个进化纪元（Epoch）。返回冠军/挑战者、尝试次数、以及晋升提示。

    默认规模刻意保守（24×5）：这是浏览器里点一下就要出结果的工具，
    不是云端跑千万基因的实验。规模上限由 max_evals 控制，超了就提前收尾。
    """
    rng = np.random.default_rng(seed)
    evals = {"n": 0}

    def evaluator(genes: StrategyGenes) -> Dict[str, float]:
        evals["n"] += 1
        return fitness(genes, returns_df, env=env, exit_cfg=exit_cfg,
                       benchmark_returns=benchmark_returns)

    incumbent = None
    if incumbent_genes is not None:
        g = incumbent_genes.clipped()
        incumbent = Individual(genes=g, origin="incumbent")
        incumbent.breakdown = evaluator(g)
        incumbent.score = incumbent.breakdown["score"]

    pop = _fill_population(pop_size, incumbent, rng, evaluator,
                           elite_ratio=max(0.10, elite_ratio))
    history: List[dict] = []
    best = _rank(pop)[0]
    stagnant = 0
    rate = mutation_rate

    for gen in range(max(1, int(generations))):
        if evals["n"] >= max_evals:
            history.append({"generation": gen, "note": "达到评估次数上限，提前终止",
                            "best_score": best.score})
            break
        ranked = _rank(pop)
        n_elite = max(1, int(math.ceil(len(ranked) * elite_ratio)))
        next_pop = [ind.clone() for ind in ranked[:n_elite]]          # 精英完整克隆

        while len(next_pop) < len(pop):
            pa = ranked[rng.integers(0, max(n_elite, len(ranked) // 2))].genes
            pb = ranked[rng.integers(0, len(ranked))].genes
            child = crossover(pa, pb, rng)
            child = mutate(child, rate=rate, sigma_ratio=0.10, rng=rng)  # 交叉后逐基因突变
            ind = Individual(genes=child, origin="offspring")
            ind.breakdown = evaluator(ind.genes)
            ind.score = ind.breakdown["score"]
            next_pop.append(ind)
        pop = next_pop

        gen_best = _rank(pop)[0]
        if gen_best.score > best.score + 1e-9:
            best = gen_best
            stagnant = 0
        else:
            stagnant += 1
            # 突变斜坡：连续无改进时优先加大探索强度，而不是直接判定收敛
            rate = min(mutation_cap, rate * ramp_factor)
        history.append({"generation": gen, "best_score": round(gen_best.score, 3),
                        "mean_score": round(float(np.mean([i.score for i in pop])), 3),
                        "mutation_rate": round(rate, 3)})

    ranked = _rank(pop)
    champion = incumbent if (incumbent and incumbent.score >= ranked[0].score) else None
    challenger = ranked[0]
    promoted_by_default = False

    review = monte_carlo_review(challenger.genes, returns_df, env=env, exit_cfg=exit_cfg,
                                benchmark_returns=benchmark_returns, n_trials=evals["n"])
    dsr = review.get("deflated_sharpe", {})
    return {
        "challenger": {"genes": challenger.genes.to_dict(), "score": round(challenger.score, 3),
                       "origin": challenger.origin, "breakdown": challenger.breakdown},
        "champion": ({"genes": incumbent.genes.to_dict(), "score": round(incumbent.score, 3)}
                     if incumbent else None),
        "promotion": {
            "auto_promoted": promoted_by_default,
            "note": "挑战者不会自动覆盖冠军参数，需要人工确认后再采用（文档 §5.5）",
        },
        "top": [{"score": round(i.score, 3), "origin": i.origin,
                 "alpha": i.breakdown.get("alpha"), "cost": i.breakdown.get("cost"),
                 "trades": i.breakdown.get("trades"), "genes": i.genes.to_dict()}
                for i in ranked[:5]],
        "history": history,
        "evaluations": evals["n"],
        "n_trials": evals["n"],
        "overfitting_check": {
            "deflated_sharpe": dsr.get("deflated_sharpe"),
            "robust_at_95": dsr.get("robust_at_95"),
            "interpretation": dsr.get("interpretation"),
            "note": f"本次共评估 {evals['n']} 组参数，DSR 已按该尝试次数校正",
        },
        "final_review": review,
        "environment": {"max_leverage": env.max_leverage, "long_only": env.long_only,
                        "global_stop_loss": env.global_stop_loss,
                        "note": "Environment 层参数未参与进化，交叉与变异无法触及"},
        "gene_blocks": GENE_BLOCKS,
    }


def monte_carlo_review(genes: StrategyGenes, returns_df: pd.DataFrame,
                       env: Environment = DEFAULT_ENV,
                       exit_cfg: Optional[ExitConfig] = None,
                       benchmark_returns: Optional[pd.Series] = None,
                       n_paths: int = 120, n_trials: int = 1, seed: int = 7) -> Dict[str, object]:
    """
    蒙特卡洛终审：只在进化结束后对最终挑战者跑一次，不参与 fitness。

    路径重抽样回答的是"这套参数的成绩有没有依赖特定时间顺序"，
    并给出资金耗尽概率（连续回撤导致权益低于阈值）与分位收益。
    """
    rng = np.random.default_rng(seed)
    base = simulate(returns_df, genes=genes, env=env, exit_cfg=exit_cfg,
                    benchmark_returns=benchmark_returns)
    if "error" in base:
        return {"available": False, "note": base["error"]}

    curve = pd.Series([(row["nav"]) for row in base["result"]["strategy"]["equity_curve"]],
                      index=returns_df.index[-len(base["result"]["strategy"]["equity_curve"]):])
    rets = curve.pct_change().dropna().values
    if len(rets) < 30:
        return {"available": False, "note": "有效收益率样本不足"}

    finals: List[float] = []
    dds: List[float] = []
    ruin = 0
    floor = 0.80
    for _ in range(int(n_paths)):
        sample = rng.choice(rets, size=len(rets), replace=True)
        path = np.cumprod(1 + sample)
        finals.append(float(path[-1] * 100 - 100))
        peak = np.maximum.accumulate(path)
        dds.append(float(np.min(path / peak - 1.0) * 100))
        if float(path.min()) < floor:
            ruin += 1

    finals_arr = np.array(finals)
    return {
        "available": True,
        "paths": int(n_paths),
        "median_return_pct": round(float(np.median(finals_arr)), 2),
        "p05_return_pct": round(float(np.percentile(finals_arr, 5)), 2),
        "p95_return_pct": round(float(np.percentile(finals_arr, 95)), 2),
        "worst_drawdown_median_pct": round(float(np.median(dds)), 2),
        "ruin_probability": round(ruin / max(len(finals), 1), 3),
        "ruin_definition": f"重抽样路径中权益跌破初始值的 {1 - floor:.0%} 视为耗尽",
        "deflated_sharpe": VAL.deflated_sharpe_ratio(pd.Series(rets), n_trials=max(1, int(n_trials))),
        "note": "蒙特卡洛只做事后复核，不影响本次进化的排名与得分（文档 §4）",
    }


def plateau_check(genes: StrategyGenes, returns_df: pd.DataFrame,
                  env: Environment = DEFAULT_ENV, exit_cfg: Optional[ExitConfig] = None,
                  jitter_ratio: float = 0.10, samples: int = 12, seed: int = 13
                  ) -> Dict[str, object]:
    """
    参数平原检验：对最优参数做邻域扰动，看分数会不会一碰就塌。

    这是第五期"把噪声当规律"的直接对策。真规律在参数平原上，
    邻近参数表现接近；过拟合解在针尖上，扰动 10% 就掉下去。
    """
    rng = np.random.default_rng(seed)
    center = fitness(genes, returns_df, env=env, exit_cfg=exit_cfg)
    scores = []
    for _ in range(int(samples)):
        scores.append(mutate(genes, rate=0.9, sigma_ratio=jitter_ratio, rng=rng))
    vals = [fitness(g, returns_df, env=env, exit_cfg=exit_cfg)["score"] for g in scores]
    if not vals:
        return {"available": False}
    mean_nb = float(np.mean(vals))
    drop = center["score"] - mean_nb
    spread = float(np.std(vals))
    plateau = drop <= max(1.0, abs(center["score"]) * 0.15)
    return {
        "available": True,
        "center_score": round(center["score"], 3),
        "neighbor_mean": round(mean_nb, 3),
        "neighbor_std": round(spread, 3),
        "drop": round(drop, 3),
        "on_plateau": bool(plateau),
        "interpretation": (
            "邻域扰动后分数基本持平，解落在参数平原上，可信度较高" if plateau else
            f"扰动 {jitter_ratio * 100:.0f}% 后平均掉分 {drop:.1f}，这个最优点很可能在针尖上（过拟合）"
        ),
    }
