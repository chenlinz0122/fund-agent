"""相关性诊断（原 quant.py E 部分）。"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

DEFAULT_CORR_THRESHOLD = 0.85


def build_returns_matrix(nav_by_code: Dict[str, pd.Series]) -> pd.DataFrame:
    """{code: nav series} -> 对齐后的日收益率 DataFrame（列=code）"""
    returns = {}
    for code, nav in nav_by_code.items():
        s = pd.Series(nav).astype(float).dropna()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        returns[code] = s.pct_change()
    df = pd.DataFrame(returns)
    return df.dropna(how="all")


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()


def high_correlation_pairs(corr_df: pd.DataFrame,
                           threshold: float = DEFAULT_CORR_THRESHOLD) -> List[Dict[str, float]]:
    """列出 |相关系数| >= threshold 的基金对（分散化预警）。"""
    pairs = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr_df.iloc[i, j]
            if pd.notna(v) and abs(v) >= threshold:
                pairs.append({"a": cols[i], "b": cols[j], "corr": round(float(v), 3)})
    return sorted(pairs, key=lambda x: abs(x["corr"]), reverse=True)


def effective_number_of_bets(corr_df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> float:
    """
    有效下注数（ENB = 1 / (w' R w)）：名义上有 N 只基金，实际相当于几个独立赌注。

    完全独立且等权时 ENB≈N；完全同涨同跌时 ENB≈1。
    用来戳穿"买了 8 只基金其实只买了 1 个方向"这类假分散。
    """
    import numpy as np
    if corr_df is None or corr_df.empty:
        return 0.0
    cols = list(corr_df.columns)
    n = len(cols)
    if weights:
        w = np.array([abs(float(weights.get(c, 0.0))) for c in cols], dtype=float)
    else:
        w = np.ones(n, dtype=float)
    total = float(np.abs(w).sum())
    if total <= 0:
        return 0.0
    w = np.abs(w) / total
    matrix = corr_df.fillna(0.0).values
    port_var = float(w @ matrix @ w)
    if port_var <= 1e-12:
        return float(n)
    return round(float(1.0 / port_var), 2)
