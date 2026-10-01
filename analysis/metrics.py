"""风险/收益指标（原 quant.py A 部分）。

约定：nav 为净值 Series（时间升序），returns 为日收益率 Series（小数）。
对外返回的百分比类指标统一为百分数（15.3 表示 15.3%）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.constants import DEFAULT_RISK_FREE, EPS, TRADING_DAYS


def to_returns(nav: pd.Series) -> pd.Series:
    """净值序列 -> 日收益率序列（保留索引以便多序列对齐）"""
    s = pd.Series(nav).astype(float).dropna()
    if len(s) < 2:
        return pd.Series(dtype=float)
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s.pct_change().dropna()


def annualized_return(nav: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """几何年化收益率（小数），按样本跨度年化。"""
    s = pd.Series(nav).astype(float).dropna()
    n = len(s)
    if n < 2:
        return 0.0
    total = s.iloc[-1] / s.iloc[0] - 1
    if total <= -1:
        return -1.0
    years = (n - 1) / trading_days
    if years <= 0:
        return 0.0
    return float((1 + total) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """年化波动率（小数）"""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（负数百分数），如 -25 表示回撤 25%。"""
    s = pd.Series(nav).astype(float).dropna()
    if len(s) < 2:
        return 0.0
    dd = s / s.cummax() - 1
    return float(dd.min() * 100)


def sharpe_ratio(returns: pd.Series, risk_free: float = DEFAULT_RISK_FREE,
                 trading_days: int = TRADING_DAYS) -> float:
    """夏普比率 = (年化收益 - 无风险) / 年化波动"""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return 0.0
    ann_ret = float(r.mean() * trading_days)
    ann_vol = annualized_volatility(r, trading_days)
    if ann_vol <= EPS:
        return 0.0
    return round((ann_ret - risk_free) / ann_vol, 3)


def sortino_ratio(returns: pd.Series, risk_free: float = DEFAULT_RISK_FREE,
                  trading_days: int = TRADING_DAYS) -> float:
    """索提诺比率：分母仅用下行波动率"""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return 0.0
    ann_ret = float(r.mean() * trading_days)
    downside = r[r < 0]
    if len(downside) == 0:
        return 0.0
    downside_vol = float(downside.std(ddof=1) * np.sqrt(trading_days))
    if downside_vol <= EPS:
        return 0.0
    return round((ann_ret - risk_free) / downside_vol, 3)


def calmar_ratio(nav: pd.Series, risk_free: float = DEFAULT_RISK_FREE,
                 trading_days: int = TRADING_DAYS) -> float:
    """卡玛比率 = (年化收益 - 无风险) / 最大回撤绝对值（同为百分数口径）"""
    s = pd.Series(nav).astype(float).dropna()
    if len(s) < 2:
        return 0.0
    ann = annualized_return(s, trading_days) * 100
    mdd = abs(max_drawdown(s))
    if mdd <= EPS:
        return 0.0
    return round((ann - risk_free * 100) / mdd, 3)


def win_rate(returns: pd.Series) -> float:
    """正收益占比（百分数）"""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) == 0:
        return 0.0
    return round(float((r > 0).sum() / len(r) * 100), 2)


def var_95(returns: pd.Series) -> float:
    """历史模拟 95% VaR（单日，百分数，负值表示亏损）"""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return 0.0
    return round(float(np.percentile(r, 5) * 100), 3)


def cvar_95(returns: pd.Series) -> float:
    """95% 条件 VaR（尾部平均损失，百分数，负值）——VaR 低估尾部时的补充。"""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return 0.0
    cutoff = np.percentile(r, 5)
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return round(float(cutoff * 100), 3)
    return round(float(tail.mean() * 100), 3)


def beta_alpha(returns: pd.Series, bench_returns: pd.Series) -> Tuple[float, float]:
    """相对基准的 Beta 与年化 Alpha（小数收益口径输入，输出百分数 alpha）。"""
    df = pd.concat([returns.rename("a"), bench_returns.rename("b")], axis=1, sort=True).dropna()
    if len(df) < 20:
        return 0.0, 0.0
    cov = np.cov(df["a"].values, df["b"].values)
    var_b = cov[1, 1]
    if var_b <= EPS:
        return 0.0, 0.0
    beta = cov[0, 1] / var_b
    a_ann = float(df["a"].mean() * TRADING_DAYS)
    b_ann = float(df["b"].mean() * TRADING_DAYS)
    alpha = a_ann - beta * b_ann
    return round(float(beta), 3), round(alpha * 100, 2)


def risk_metrics(nav: pd.Series, bench_nav: Optional[pd.Series] = None,
                 risk_free: float = DEFAULT_RISK_FREE) -> Dict[str, float]:
    """
    由净值序列计算一整套风险/收益指标。

    返回字段：annual_return_pct, total_return_pct, volatility_pct, max_drawdown_pct,
    sharpe, sortino, calmar, win_rate_pct, var_95_pct, cvar_95_pct, latest_nav,
    data_points；提供 bench_nav 时追加 beta, alpha_pct。
    """
    s = pd.Series(nav).astype(float).dropna()
    if len(s) >= 2 and not s.index.is_monotonic_increasing:
        s = s.sort_index()
    if len(s) < 2:
        return {"error": "数据点不足", "data_points": int(len(s))}
    r = to_returns(s)
    m = {
        "annual_return_pct": round(annualized_return(s) * 100, 2),
        "total_return_pct": float(round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2)),
        "volatility_pct": round(annualized_volatility(r) * 100, 2),
        "max_drawdown_pct": round(max_drawdown(s), 2),
        "sharpe": sharpe_ratio(r, risk_free),
        "sortino": sortino_ratio(r, risk_free),
        "calmar": calmar_ratio(s, risk_free),
        "win_rate_pct": win_rate(r),
        "var_95_pct": var_95(r),
        "cvar_95_pct": cvar_95(r),
        "latest_nav": round(float(s.iloc[-1]), 4),
        "data_points": int(len(s)),
    }
    if bench_nav is not None:
        b = pd.Series(bench_nav).astype(float).dropna()
        beta, alpha = beta_alpha(r, to_returns(b))
        m["beta"] = beta
        m["alpha_pct"] = alpha
    return m
