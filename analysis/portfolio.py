"""组合优化（原 quant.py B 部分）+ 仓位护栏。

这里体现视频里那条"残酷数学真相"的工程化落地：
  1) 做空在数学期望上为负 —— 公募基金无法做空，所以权重必须非负，
     出现负权重说明闭式解已经不适用于本场景，必须裁剪并如实标注；
  2) 杠杆放大方差、不改变期望 —— 总仓位硬约束 <= 100%，多余部分留现金；
  3) 波动率才是仓位的因变量 —— 用目标波动率反推仓位，而不是凭感觉满仓。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.constants import DEFAULT_RISK_FREE, EPS, TRADING_DAYS


# ========== 协方差处理 ==========
def _regularized_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2
    n = cov.shape[0]
    trace = np.trace(cov) / max(n, 1)
    ridge = max(trace * 1e-4, EPS)
    return cov + np.eye(n) * ridge


def _normalize_long_only(w: np.ndarray) -> np.ndarray:
    """裁剪为非负并归一化；全负/全零退化为等权。"""
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= EPS:
        return np.ones_like(w) / len(w)
    return w / s


def weights_dict(weights: np.ndarray, cols: List[str]) -> Dict[str, float]:
    w = np.asarray(weights, dtype=float)
    if w.sum() != 0:
        w = w / w.sum()
    return {c: round(float(w[i] * 100), 2) for i, c in enumerate(cols)}


def raw_weights_pct(weights, cols: List[str]) -> Dict[str, float]:
    """
    按"原始占比"输出百分比，不做归一化。

    weights_dict 会把权重重新除以总和，这在目标波动率缩放的场景里是错的：
    缩放后总仓位低于 100% 是刻意留现金，归一化等于偷偷加满仓。
    """
    w = np.asarray(weights, dtype=float)
    return {c: round(float(w[i]) * 100, 2) for i, c in enumerate(cols)}


# ========== 经典解 ==========
def min_variance_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """全局最小方差组合（sum=1）。"""
    cov = _regularized_cov(cov)
    n = cov.shape[0]
    ones = np.ones(n)
    inv_1 = np.linalg.solve(cov, ones)
    denom = ones @ inv_1
    if abs(denom) <= EPS:
        return ones / n
    w = inv_1 / denom
    return _normalize_long_only(w) if long_only else w


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray,
                       risk_free: float = DEFAULT_RISK_FREE,
                       long_only: bool = True) -> np.ndarray:
    """最大夏普（切线组合）。"""
    mu = np.asarray(mu, dtype=float)
    cov = _regularized_cov(cov)
    n = cov.shape[0]
    w = np.linalg.solve(cov, mu - risk_free)
    if long_only:
        return _normalize_long_only(w)
    s = w.sum()
    if abs(s) <= EPS:
        return np.ones(n) / n
    return w / s


def risk_parity_weights(cov: np.ndarray, iters: int = 500) -> np.ndarray:
    """风险平价：乘法迭代，无需外部优化器。"""
    cov = _regularized_cov(cov)
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        marginal = np.clip(cov @ w, EPS, None)
        rc = w * marginal
        target = rc.mean()
        w_new = np.clip(w * (target / rc), 0, None)
        if w_new.sum() <= EPS:
            break
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new
    return w


def efficient_frontier(mu: np.ndarray, cov: np.ndarray,
                       risk_free: float = DEFAULT_RISK_FREE,
                       n_points: int = 24) -> List[Dict[str, object]]:
    """经典均值-方差有效前沿（Merton 闭式解；展示用的权重仍裁剪为 long-only）。"""
    mu = np.asarray(mu, dtype=float)
    cov = _regularized_cov(cov)
    n = cov.shape[0]
    ones = np.ones(n)
    inv_1 = np.linalg.solve(cov, ones)
    inv_mu = np.linalg.solve(cov, mu)
    A = float(mu @ inv_1)
    B = float(mu @ inv_mu)
    C = float(ones @ inv_1)
    D = B * C - A * A
    if abs(D) <= EPS or abs(C) <= EPS:
        return []
    r_min_var = A / C
    r_max = float(np.max(mu))
    if r_max <= r_min_var:
        r_max = r_min_var * 1.5 + 0.05
    points = []
    for rp in np.linspace(r_min_var, r_max, n_points):
        w = ((C - A * rp) * inv_1 + (B * rp - A) * inv_mu) / D
        var_p = (C * rp ** 2 - 2 * A * rp + B) / D
        vol_p = np.sqrt(max(var_p, 0))
        points.append({
            "volatility_pct": float(round(float(vol_p) * 100, 2)),
            "return_pct": float(round(float(rp) * 100, 2)),
            "weights": [float(round(float(x) * 100, 2)) for x in _normalize_long_only(w)],
        })
    return points


def _portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                     risk_free: float = DEFAULT_RISK_FREE) -> Tuple[float, float, float]:
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0)))
    sr = (ret - risk_free) / vol if vol > EPS else 0.0
    return ret, vol, sr


# ========== 护栏：不加杠杆、不做空 ==========
def leverage_guard(weights, max_gross_exposure: float = 1.0) -> Dict[str, object]:
    """
    检查权重是否违反 long-only / 无杠杆约束。

    weights 可以是 {code: pct} 或 ndarray（百分比或小数都能识别：绝对值 >1.5 视为百分比）。
    """
    if isinstance(weights, dict):
        values = np.array(list(weights.values()), dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
    if values.size == 0:
        return {"ok": True, "gross_exposure_pct": 0.0, "short_weights": [], "note": "空组合"}

    is_pct = float(np.abs(values).max()) > 1.5
    frac = values / 100.0 if is_pct else values

    gross = float(np.abs(frac).sum())
    shorts = [round(float(v) * 100, 4) for v in frac if v < -1e-9]
    leveraged = gross > max_gross_exposure + 1e-6
    ok = (not shorts) and (not leveraged)

    notes = []
    if shorts:
        notes.append(f"检测到 {len(shorts)} 个负权重：公募基金无法做空，做空在期望上为负，已应裁剪为 0")
    if leveraged:
        notes.append(f"总敞口 {gross * 100:.1f}% 超过上限 {max_gross_exposure * 100:.0f}%：杠杆只放大方差，不改变期望")
    if not notes:
        notes.append("满足 long-only、无杠杆约束")

    return {
        "ok": ok,
        "gross_exposure_pct": round(gross * 100, 2),
        "short_weights": shorts,
        "notes": notes,
    }


def vol_target_weights(returns_df: pd.DataFrame, weights=None,
                       target_vol: float = 0.10,
                       max_gross_exposure: float = 1.0,
                       lookback: Optional[int] = None,
                       trading_days: int = TRADING_DAYS) -> Dict[str, object]:
    """
    目标波动率仓位：把组合已实现波动率缩放到 target_vol，剩余部分留现金。

    缩放系数上限受 max_gross_exposure 约束（默认 1.0），
    因此"波动率低于目标"时不会自动加杠杆，只会保留现金 —— 这正是刻意的。
    """
    df = returns_df.dropna(axis=1, how="all").dropna()
    cols = list(df.columns)
    if not cols or len(df) < 2:
        return {"error": "无足够数据估计波动率"}

    if weights is None:
        w = np.ones(len(cols)) / len(cols)
    elif isinstance(weights, dict):
        w = np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > EPS else np.ones(len(cols)) / len(cols)
    else:
        w = np.asarray(weights, dtype=float)

    window = int(lookback or min(len(df), 63))
    realized = float(np.sqrt(max(w @ (df.tail(window).cov().values) @ w, 0)) * np.sqrt(trading_days))
    if realized <= EPS:
        scale = 1.0
    else:
        scale = min(target_vol / realized, max_gross_exposure)

    invested = w * scale
    cash = float(max(1.0 - float(invested.sum()), 0.0))
    invested_pct = raw_weights_pct(invested, cols)
    return {
        "weights": invested_pct,
        "cash_weight_pct": round(cash * 100, 2),
        "scale": round(float(scale), 4),
        "realized_volatility_pct": round(realized * 100, 2),
        "target_volatility_pct": round(target_vol * 100, 2),
        "lookback_days": window,
        "leveraged": bool(scale > 1.0 + 1e-6),
        "guard": leverage_guard(invested_pct, max_gross_exposure),
    }


# ========== 主入口 ==========
def optimize_portfolio(returns_df: pd.DataFrame,
                       risk_free: float = DEFAULT_RISK_FREE,
                       target_vol: Optional[float] = None,
                       max_gross_exposure: float = 1.0) -> Dict[str, object]:
    """
    输入对齐后的日收益率 DataFrame（列=基金代码）。

    输出：最小方差 / 最大夏普 / 风险平价 三组权重 + 预期收益波动夏普 + 有效前沿，
    以及每组的 long-only/无杠杆校验结果；给定 target_vol 时追加目标波动率仓位。
    """
    df = returns_df.dropna(axis=1, how="all").dropna()
    cols = list(df.columns)
    if len(cols) < 2:
        return {"error": "至少需要 2 只基金，且时间区间需对齐", "funds": cols}

    mu = df.mean().values * TRADING_DAYS
    cov = df.cov().values * TRADING_DAYS

    def block(name, w):
        r, v, s = _portfolio_stats(w, mu, cov, risk_free)
        wd = weights_dict(w, cols)
        return {
            "name": name,
            "weights": wd,
            "expected_return_pct": round(r * 100, 2),
            "expected_volatility_pct": round(v * 100, 2),
            "expected_sharpe": round(s, 3),
            "guard": leverage_guard(wd, max_gross_exposure),
        }

    w_minvar = min_variance_weights(cov)
    out = {
        "funds": cols,
        "sample_days": int(len(df)),
        "results": [
            block("max_sharpe", max_sharpe_weights(mu, cov, risk_free)),
            block("min_variance", w_minvar),
            block("risk_parity", risk_parity_weights(cov)),
        ],
        "efficient_frontier": efficient_frontier(mu, cov, risk_free),
    }
    if target_vol is not None:
        out["vol_targeted"] = vol_target_weights(df, None, target_vol,
                                                 max_gross_exposure=max_gross_exposure)
    return out
