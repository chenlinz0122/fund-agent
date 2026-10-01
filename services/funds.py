"""单只基金详情用例：净值 + 风险指标 + 费后期望。"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from analysis import expectancy as exp
from analysis import metrics as M
from config import Settings
from core.errors import InvalidInput, SourceUnavailable
from sources.repository import FundRepository

# 前端依赖的字段名（保持与旧版一致，新增字段只做追加）
_METRIC_RENAME = {
    "total_return_pct": "total_return_1y",
    "annual_return_pct": "annual_return",
    "volatility_pct": "volatility",
    "sharpe": "sharpe_ratio",
    "sortino": "sortino",
    "calmar": "calmar",
    "max_drawdown_pct": "max_drawdown",
    "win_rate_pct": "win_rate",
    "beta": "beta",
    "latest_nav": "latest_nav",
    "data_points": "data_points",
}


def fund_detail(code: str, repo: FundRepository, settings: Settings,
                days: int = 365, force: bool = False) -> dict:
    """获取单只基金详情（含历史净值、风险指标、费后期望）。"""
    code = str(code).strip()
    if not code.isdigit() or len(code) != 6:
        raise InvalidInput("基金代码格式错误，应为 6 位数字")

    if not force:
        cached = repo.cached_detail(code)
        if cached is not None:
            return cached

    info = repo.basic_info(code)
    try:
        nav = repo.nav_history(code, days=days)
    except SourceUnavailable as e:
        raise

    detail: Dict[str, object] = {"code": code, "info": info, "history": [], "metrics": {}}
    if len(nav) < 2:
        if repo.nav_source_blocked():
            raise SourceUnavailable("天天基金净值接口正在限流冷却中，请 1 分钟后重试")
        detail["warning"] = "未取到净值历史，无法计算风险指标（可能是代码不存在或网络受限）"
        return detail

    detail["history"] = [
        {"date": d.strftime("%Y-%m-%d"), "nav": round(float(v), 4)} for d, v in nav.items()
    ]

    bench = repo.benchmark(days=days)
    b_nav = bench.reindex(nav.index, method="nearest") if bench is not None and len(bench) > 1 else None
    raw = M.risk_metrics(nav, bench_nav=b_nav, risk_free=settings.analysis.risk_free)
    detail["metrics"] = {new: raw.get(old) for old, new in _METRIC_RENAME.items()}
    detail["metrics_extra"] = {k: raw.get(k) for k in ("var_95_pct", "cvar_95_pct", "alpha_pct")}

    fund_type = info.get("基金类型") or info.get("type") or None
    detail["expectancy"] = exp.expectancy_report(
        raw, fund_type=fund_type, returns=M.to_returns(nav),
        fee_cfg=settings.fees, analysis_cfg=settings.analysis,
    )
    repo.cache_detail(code, detail)
    return detail
