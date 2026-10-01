"""策略模拟与参数进化的用例编排。

一条硬性安全规则在这一层就把住：**客户端提交的参数只能落在 StrategyGenes /
ExitConfig 白名单里**，Environment（杠杆上限、long-only、全局熔断、费用下限）
不接受外部覆盖。这样"进化出杠杆策略"在 API 边界上也不可能发生。
"""
from __future__ import annotations

from dataclasses import fields
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.abg import ABGParams
from analysis.risk_rules import DEFAULT_EXIT, ExitConfig, describe_rules, evaluate_exit
from analysis.search import evolve, plateau_check
from analysis.signals import (
    DEFAULT_ENV, Environment, StrategyGenes, exposure_from_state, read_state,
    season_multiplier, target_exposure,
)
from analysis.simulation import simulate
from config import Settings
from core.errors import InvalidInput, SourceUnavailable
from sources.repository import FundRepository

GENE_FIELDS = {f.name for f in fields(StrategyGenes)} - {"abg"}
ABG_FIELDS = {f.name for f in fields(ABGParams)}
EXIT_FIELDS = {f.name for f in fields(ExitConfig)}


def build_genes(payload: Optional[dict]) -> StrategyGenes:
    """从请求体构造基因，忽略一切不在白名单里的键（尤其是环境层）。"""
    payload = payload or {}
    unknown = [k for k in payload if k not in GENE_FIELDS and k != "abg"]
    kwargs = {k: v for k, v in payload.items() if k in GENE_FIELDS}
    genes = StrategyGenes(**kwargs)
    abg_payload = payload.get("abg") or {}
    abg_kwargs = {k: v for k, v in abg_payload.items() if k in ABG_FIELDS}
    if abg_kwargs:
        genes.abg = ABGParams(**abg_kwargs)
    genes = genes.clipped()
    genes._ignored_keys = unknown          # 附带说明用，不参与计算
    return genes


def build_exit(payload: Optional[dict]) -> ExitConfig:
    payload = payload or {}
    kwargs = {k: v for k, v in payload.items() if k in EXIT_FIELDS}
    return ExitConfig(**kwargs).clipped()


def ignored_keys(genes: StrategyGenes) -> List[str]:
    return list(getattr(genes, "_ignored_keys", []) or [])


def run_simulation(repo: FundRepository, settings: Settings, codes: List[str], days: int = 365,
                   genes_payload: Optional[dict] = None, exit_payload: Optional[dict] = None,
                   weights: Optional[Dict[str, float]] = None,
                   season: Optional[str] = None) -> Dict[str, object]:
    from services.portfolio import clean_codes
    codes = clean_codes(codes)
    returns_df, nav_by_code = repo.require_aligned(codes, days)
    genes = build_genes(genes_payload)
    cfg = build_exit(exit_payload)
    bench = repo.benchmark(days=days)
    bench_rets = None
    if bench is not None and len(bench) > 1:
        from analysis.metrics import to_returns
        bench_rets = to_returns(bench)

    result = simulate(returns_df, genes=genes, env=DEFAULT_ENV, exit_cfg=cfg,
                      weights=weights, benchmark_returns=bench_rets, season=season)
    if "error" in result:
        raise InvalidInput(result["error"])

    result["policy"] = describe_rules(genes, DEFAULT_ENV, cfg)
    result["current_stance"] = _current_stance(returns_df, genes, cfg)
    if ignored_keys(genes):
        result["ignored_params"] = ignored_keys(genes)
        result["ignored_reason"] = ("这些字段不在可进化参数盒内（环境层红线与未知字段不可覆盖）："
                                    + ", ".join(ignored_keys(genes)))
    return result


def _current_stance(returns_df: pd.DataFrame, genes: StrategyGenes,
                    cfg: ExitConfig) -> Dict[str, object]:
    """把"现在该持有多少仓位"讲清楚：天平明细 + 出场判定。"""
    equal = np.ones(len(returns_df.columns)) / max(len(returns_df.columns), 1)
    base_nav = pd.Series(np.cumprod(1 + (returns_df.values @ equal)), index=returns_df.index)
    state = read_state(base_nav, genes)
    stance = exposure_from_state(state, genes, DEFAULT_ENV, season_multiplier(None))
    exit_now = evaluate_exit(base_nav, genes=genes, env=DEFAULT_ENV, cfg=cfg)
    return {"state": state, "target_exposure": stance, "exit_check": exit_now}


def run_evolution(repo: FundRepository, settings: Settings, codes: List[str], days: int = 365,
                  pop_size: int = 16, generations: int = 4, seed: int = 2026,
                  incumbent_payload: Optional[dict] = None,
                  exit_payload: Optional[dict] = None,
                  check_plateau: bool = True) -> Dict[str, object]:
    from services.portfolio import clean_codes
    codes = clean_codes(codes)
    returns_df, _ = repo.require_aligned(codes, days)
    genes = build_genes(incumbent_payload) if incumbent_payload else None
    cfg = build_exit(exit_payload)

    # 评估次数上限：pop_size*(1+generations) 是自然需求，再套一层 300 的硬顶保护服务进程
    max_evals = min(int(pop_size) * (1 + int(generations)), 300)
    report = evolve(returns_df, env=DEFAULT_ENV, exit_cfg=cfg, pop_size=pop_size,
                    generations=generations, seed=seed, incumbent_genes=genes,
                    max_evals=max_evals)
    best = report["challenger"]["genes"]
    best_genes = build_genes(best)
    if check_plateau:
        report["plateau"] = plateau_check(best_genes, returns_df, env=DEFAULT_ENV, exit_cfg=cfg)
    report["requested"] = {"codes": codes, "days": days, "pop_size": pop_size,
                           "generations": generations, "seed": seed}
    report["cost_note"] = (
        f"本次共评估 {report['evaluations']} 组参数。尝试次数越多，"
        f"最好的那个分数越可能是噪声——DSR 已按此校正，请看 robust_at_95 而不是看最高分。")
    return report


def policy_snapshot(codes: List[str], repo: FundRepository, days: int = 365,
                    genes_payload: Optional[dict] = None) -> Dict[str, object]:
    """每只基金当前的状态与建议敞口（不下单，只给风险档位）。"""
    from services.portfolio import clean_codes
    codes = clean_codes(codes, min_count=1)
    genes = build_genes(genes_payload)
    rows = []
    for code in codes:
        nav = repo.nav_history(code, days=days)
        if len(nav) < genes.min_history:
            rows.append({"code": code, "available": False,
                         "note": f"净值样本 {len(nav)} 天，不足以判断状态"})
            continue
        state = read_state(nav, genes)
        stance = exposure_from_state(state, genes, DEFAULT_ENV, season_multiplier(None))
        rows.append({"code": code, "available": True,
                     "nav_points": int(len(nav)),
                     "state": state, "target_exposure": stance})

    if rows and all(not r["available"] for r in rows) and repo.nav_source_blocked():
        raise SourceUnavailable("天天基金净值接口正在限流冷却中，请 1 分钟后重试（不是这些基金没有数据）")
    return {"codes": codes, "days": days, "rows": rows,
            "policy": describe_rules(genes, DEFAULT_ENV, DEFAULT_EXIT),
            "disclaimer": "敞口档位是风险预算的表达，不是收益预测，也不构成投资建议"}
