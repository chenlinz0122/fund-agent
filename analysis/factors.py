"""多因子综合打分（原 quant.py D 部分）。

Z-score 标准化后加权合成。注意：打分只描述"过去这段时间的统计特征"，
它是否对未来有预测力，要用 analysis.validation.walk_forward 检验，
不要把打分本身当成期望收益为正的证据。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from analysis.constants import EPS

# 因子方向：True 越大越好；False 越小越好（内部已取负，统一按越大越好处理）
DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "annual_return_pct": 0.30,
    "sharpe": 0.25,
    "calmar": 0.15,
    "win_rate_pct": 0.10,
    "volatility_pct": 0.10,
    "max_drawdown": 0.10,
}

# 可选因子：只有当输入指标里真的存在、且权重表里显式给了权重时才参与打分
OPTIONAL_FACTORS = ("net_annual_return_pct", "cvar_95_pct", "sortino")


def _extract_row(m: Dict[str, float], include: List[str]) -> Dict[str, float]:
    row = {
        "annual_return_pct": m.get("annual_return_pct", 0) or 0,
        "sharpe": m.get("sharpe", 0) or 0,
        "calmar": m.get("calmar", 0) or 0,
        "win_rate_pct": m.get("win_rate_pct", 0) or 0,
        "volatility_pct": -(m.get("volatility_pct", 0) or 0),      # 波动小更好
        "max_drawdown": -abs(m.get("max_drawdown_pct", 0) or 0),   # 回撤小更好
    }
    if "net_annual_return_pct" in include:
        row["net_annual_return_pct"] = m.get("net_annual_return_pct", 0) or 0
    if "cvar_95_pct" in include:
        row["cvar_95_pct"] = -(abs(m.get("cvar_95_pct", 0) or 0))  # 尾部损失小更好
    if "sortino" in include:
        row["sortino"] = m.get("sortino", 0) or 0
    return row


def multi_factor_score(metrics_list: List[Dict[str, float]],
                       weights_map: Optional[Dict[str, float]] = None) -> List[Dict[str, object]]:
    """
    对一组基金的指标做 Z-score 标准化并按方向加权，合成 0-100 综合评分（越高越好）。

    metrics_list 每项需含 code + 指标字段（analysis.metrics.risk_metrics 的输出即可）。
    """
    if not metrics_list:
        return []

    weights_map = weights_map or dict(DEFAULT_FACTOR_WEIGHTS)
    include = [f for f in OPTIONAL_FACTORS if f in weights_map]

    rows = []
    for m in metrics_list:
        if "error" in m:
            continue
        rows.append((m.get("code"), _extract_row(m, include)))
    if not rows:
        return []

    factors = list(rows[0][1].keys())
    matrix = np.array([[r[f] for f in factors] for _, r in rows], dtype=float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds[stds <= EPS] = 1.0
    z = (matrix - means) / stds

    wvec = np.array([weights_map.get(f, DEFAULT_FACTOR_WEIGHTS.get(f, 0.0)) for f in factors], dtype=float)
    if wvec.sum() <= EPS:
        wvec = np.ones(len(factors)) / len(factors)
    else:
        wvec = wvec / wvec.sum()

    raw = z @ wvec
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    scores = [(raw[i] - lo) / span * 100 if span > EPS else 50.0 for i in range(len(raw))]

    out = []
    for i, (code, _) in enumerate(rows):
        out.append({
            "code": code,
            "score": float(round(scores[i], 2)),
            "factor_zscores": {k: float(round(float(z[i, j]), 3)) for j, k in enumerate(factors)},
        })
    return sorted(out, key=lambda x: x["score"], reverse=True)
