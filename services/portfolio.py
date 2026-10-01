"""组合分析用例：指标、相关性、优化、回测、期望值、样本外稳健性。

原先这些逻辑直接写在 FastAPI 路由函数里，导致：
  - 无法在没有 HTTP 的情况下复现一个计算结果；
  - 每个路由都要重复 try/except 与数据对齐。
现在路由只做"解析请求 -> 调这里 -> 返回"。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from analysis.backtest import backtest as bt_backtest
from analysis import correlation as CORR
from analysis import expectancy as exp
from analysis import factors as FAC
from analysis import metrics as M
from analysis import portfolio as PF
from analysis import validation as VAL
from config import Settings
from core.errors import InsufficientData, InvalidInput
from sources.repository import FundRepository


def clean_codes(codes: List[str], min_count: int = 2) -> List[str]:
    """去重并只保留 6 位数字基金代码。"""
    out: List[str] = []
    for c in codes or []:
        c = str(c).strip()
        if re.match(r"^\d{6}$", c) and c not in out:
            out.append(c)
    if len(out) < min_count:
        raise InvalidInput(f"请至少提供 {min_count} 只有效的 6 位基金代码")
    return out


def _with_bench(nav: pd.Series, bench: pd.Series) -> Optional[pd.Series]:
    if bench is None or len(bench) < 2 or nav is None or len(nav) == 0:
        return None
    return bench.reindex(nav.index, method="nearest")


def analyze(repo: FundRepository, settings: Settings, codes: List[str],
            days: int = 365) -> Dict[str, object]:
    """A 风险指标 + D 多因子打分 + E 相关性诊断（含假分散检测）。"""
    codes = clean_codes(codes)
    returns_df, nav_by_code = repo.require_aligned(codes, days)
    valid = list(returns_df.columns)
    bench = repo.benchmark(days=days)
    cfg = settings.analysis

    per_fund: List[dict] = []
    for code in valid:
        nav = nav_by_code.get(code, pd.Series(dtype=float))
        raw = M.risk_metrics(nav, bench_nav=_with_bench(nav, bench), risk_free=cfg.risk_free)
        raw["code"] = code
        per_fund.append(raw)

    corr_df = CORR.correlation_matrix(returns_df)
    correlation = {
        a: {b: (None if pd.isna(corr_df.loc[a, b]) else float(round(corr_df.loc[a, b], 3)))
            for b in corr_df.columns}
        for a in corr_df.columns
    }
    scores = FAC.multi_factor_score(per_fund, settings.factor_weights)
    score_map = {s["code"]: s["score"] for s in scores}
    for m in per_fund:
        m["composite_score"] = score_map.get(m["code"])

    equal_weights = {c: 100.0 / len(valid) for c in valid}
    return {
        "funds": valid,
        "sample_days": int(returns_df.shape[0]),
        "window_days": days,
        "metrics": per_fund,
        "correlation": correlation,
        "high_correlation_pairs": CORR.high_correlation_pairs(corr_df, cfg.corr_warn_threshold),
        "effective_number_of_bets": CORR.effective_number_of_bets(corr_df, equal_weights),
        "risk_contributions": risk_contributions(returns_df, equal_weights),
        "ranking": scores,
        "benchmark": f"{settings.source.benchmark_name}({settings.source.benchmark_symbol})"
                     if bench is not None and len(bench) > 1 else None,
    }


def risk_contributions(returns_df: pd.DataFrame, weights: Dict[str, float],
                       trading_days: int = 252) -> Dict[str, float]:
    """等权/给定权重下各基金对组合波动的贡献占比 —— 谁在真正决定风险。"""
    import numpy as np
    cols = list(returns_df.columns)
    w = np.array([float(weights.get(c, 0.0)) for c in cols]) / 100.0
    cov = returns_df.cov().values * trading_days
    port_vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
    if port_vol <= 1e-12:
        return {c: round(100.0 / len(cols), 2) for c in cols}
    rc = w * (cov @ w) / port_vol
    total = float(np.abs(rc).sum()) or 1.0
    return {c: round(float(rc[i]) / total * 100, 2) for i, c in enumerate(cols)}


def optimize(repo: FundRepository, settings: Settings, codes: List[str], days: int = 365,
             target_vol: Optional[float] = None) -> Dict[str, object]:
    """B 组合优化 + 无杠杆/不做空护栏 + 目标波动率仓位。"""
    codes = clean_codes(codes)
    returns_df, _ = repo.require_aligned(codes, days)
    cfg = settings.analysis
    result = PF.optimize_portfolio(
        returns_df, risk_free=cfg.risk_free,
        target_vol=target_vol if target_vol is not None else cfg.target_volatility,
        max_gross_exposure=cfg.max_gross_exposure,
    )
    if "error" in result:
        raise InsufficientData(result["error"])
    return result


def run_backtest(repo: FundRepository, settings: Settings, codes: List[str], days: int = 365,
                 weights: Optional[Dict[str, float]] = None, rebalance: str = "monthly",
                 cost_pct: Optional[float] = None) -> Dict[str, object]:
    """C 组合回测（含换手成本）。"""
    codes = clean_codes(codes)
    returns_df, _ = repo.require_aligned(codes, days)
    bench = repo.benchmark(days=days)
    bench_rets = M.to_returns(bench) if bench is not None and len(bench) > 1 else None
    result = bt_backtest(
        returns_df, weights=weights, rebalance=rebalance, bench_returns=bench_rets,
        cost_pct=cost_pct, risk_free=settings.analysis.risk_free,
    )
    if "error" in result:
        raise InsufficientData(result["error"])
    return result


def robustness(repo: FundRepository, settings: Settings, codes: List[str], days: int = 1095,
               train_days: int = 252, test_days: int = 63, folds: int = 4,
               n_trials: int = 1) -> Dict[str, object]:
    """
    反过拟合体检：排名规则在样本外还有没有预测力，最优夏普是不是试出来的。

    需要较长历史（默认 3 年），数据不足时明确返回 available=False 而不是硬算。
    """
    codes = clean_codes(codes, min_count=3)
    nav_by_code = {c: repo.nav_history(c, days=days) for c in codes}
    nav_by_code = {c: s for c, s in nav_by_code.items() if len(s) > 30}
    if len(nav_by_code) < 3:
        raise InsufficientData("至少需要 3 只基金各有 30 天以上历史")

    wf = VAL.walk_forward(nav_by_code, train_days=train_days, test_days=test_days, folds=folds)

    per_fund = []
    for code, nav in nav_by_code.items():
        rets = M.to_returns(nav)
        raw = M.risk_metrics(nav, risk_free=settings.analysis.risk_free)
        dsr = VAL.deflated_sharpe_ratio(rets, n_trials=n_trials,
                                       risk_free_annual=settings.analysis.risk_free)
        per_fund.append({
            "code": code,
            "sharpe": raw.get("sharpe"),
            "deflated_sharpe": dsr.get("deflated_sharpe"),
            "robust_at_95": dsr.get("robust_at_95"),
            "observations": dsr.get("observations"),
            "interpretation": dsr.get("interpretation"),
        })
    per_fund.sort(key=lambda x: (x.get("deflated_sharpe") is not None,
                                 x.get("deflated_sharpe") or -1), reverse=True)

    return {
        "walk_forward": wf,
        "per_fund_deflated_sharpe": per_fund,
        "n_trials": int(n_trials),
        "note": "样本外 Rank IC 接近 0 时，说明历史排名对未来没有预测力；"
                "DSR 随尝试的参数个数增加而下降，这是「搜索出来的漂亮回测」的真实折价。",
    }


def expectancy_batch(repo: FundRepository, settings: Settings, codes: List[str],
                     days: int = 365, round_trips_per_year: Optional[float] = None,
                     fund_types: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """一组基金的费后期望对照表（毛收益 vs 净收益 vs 净夏普）。"""
    codes = clean_codes(codes)
    rows = []
    for code in codes:
        nav = repo.nav_history(code, days=days)
        if len(nav) < 2:
            rows.append({"code": code, "error": "净值数据不足"})
            continue
        raw = M.risk_metrics(nav, risk_free=settings.analysis.risk_free)
        report = exp.expectancy_report(
            raw, fund_type=(fund_types or {}).get(code), returns=M.to_returns(nav),
            round_trips_per_year=round_trips_per_year,
            fee_cfg=settings.fees, analysis_cfg=settings.analysis,
        )
        rows.append({
            "code": code,
            "gross_annual_return_pct": report["gross_annual_return_pct"],
            "annualized_fee_drag_pct": report["annualized_fee_drag_pct"],
            "net_annual_return_pct": report["net_annual_return_pct"],
            "net_sharpe": report["net_sharpe"],
            "positive_expectancy": report["positive_expectancy"],
            "risk_points": report["risk_of_ruin_notes"],
            "verdict": report["verdict"],
        })
    rows.sort(key=lambda r: r.get("net_annual_return_pct") if r.get("net_annual_return_pct") is not None else -999,
              reverse=True)
    return {
        "rows": rows,
        "assumption": "费率采用主流互联网代销平台的常见假设值，真实费率以基金法律文件为准",
        "round_trips_per_year": round_trips_per_year if round_trips_per_year is not None
        else settings.fees.default_round_trips_per_year,
    }
