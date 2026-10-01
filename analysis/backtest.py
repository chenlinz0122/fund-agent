"""组合回测（原 quant.py C 部分）。

顺手修掉一个语义 bug：原实现里 rebalance="monthly" 和 "daily" 走的是同一条
"每天固定权重"分支，等于把月度再平衡偷偷当成了每日再平衡 —— 这会系统性
高估再平衡收益、并且完全忽略换手成本。现在 monthly 是真正的月末再平衡，
且可以按换手率扣除申赎费用（费用是确定性负期望，不该是 0）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis import metrics as M
from analysis.constants import DEFAULT_RISK_FREE, EPS, TRADING_DAYS
from analysis.portfolio import weights_dict


def _resolve_weights(cols: List[str], weights) -> np.ndarray:
    if weights is None:
        return np.ones(len(cols)) / len(cols)
    if isinstance(weights, dict):
        w = np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0, None)          # long-only：负权重一律裁剪
    if w.sum() <= EPS:
        return np.ones(len(cols)) / len(cols)
    return w / w.sum()


def _month_keys(index: pd.DatetimeIndex) -> List[int]:
    return [index[i].year * 100 + index[i].month for i in range(len(index))]


def backtest(returns_df: pd.DataFrame, weights=None, rebalance: str = "monthly",
             bench_returns: Optional[pd.Series] = None,
             cost_pct: Optional[float] = None,
             risk_free: float = DEFAULT_RISK_FREE,
             trading_days: int = TRADING_DAYS) -> Dict[str, object]:
    """
    给定对齐后的日收益率表与权重，回测组合净值曲线。

    rebalance: 'none' 买入持有 / 'monthly' 月末再平衡 / 'daily' 每日恢复目标权重
    cost_pct : 单边换手成本（百分数，如 0.3 表示 0.3%）。None 表示不计成本。
    """
    df = returns_df.dropna(axis=1, how="all").dropna()
    cols = list(df.columns)
    if not cols or len(df) == 0:
        return {"error": "无可用对齐数据"}

    w = _resolve_weights(cols, weights)
    rets = df.values                     # T x N
    dates = df.index
    cost_rate = (float(cost_pct) / 100.0) if cost_pct else 0.0

    total_cost = 0.0
    total_turnover = 0.0
    values: List[float] = []

    if rebalance == "daily":
        equity = 1.0
        for t in range(rets.shape[0]):
            equity *= (1.0 + float(rets[t] @ w))
            values.append(equity)

    elif rebalance == "monthly":
        keys = _month_keys(dates)
        holdings = w.copy()              # 每 1 元初始资金对应的持仓市值
        equity = 1.0
        prev_month = None
        for t in range(rets.shape[0]):
            if prev_month is not None and keys[t] != prev_month:
                # 月末/月初边界：把持仓拉回目标权重，并按换手计费
                before = holdings.copy()
                holdings = equity * w
                turnover = float(np.abs(holdings - before).sum()) / max(equity, EPS)
                total_turnover += turnover
                if cost_rate:
                    fee = equity * turnover * cost_rate
                    equity -= fee
                    holdings *= (equity / max(equity + fee, EPS))
                    total_cost += fee
            prev_month = keys[t]
            holdings = holdings * (1.0 + rets[t])
            equity = float(holdings.sum())
            values.append(equity)
        values = [v / values[0] for v in values]

    else:  # none：买入持有，权重随涨跌漂移
        holdings = w.copy()
        for t in range(rets.shape[0]):
            holdings = holdings * (1.0 + rets[t])
            values.append(float(holdings.sum()))
        values = [v / values[0] for v in values]

    curve = pd.Series(values, index=dates, dtype=float)
    stats = M.risk_metrics(curve, risk_free=risk_free)

    out: Dict[str, object] = {
        "weights": weights_dict(w, cols),
        "rebalance": rebalance,
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "nav": round(float(v), 4)}
            for d, v in curve.items()
        ],
        "metrics": stats,
    }
    if cost_pct:
        out["cost_pct"] = float(cost_pct)
        out["total_turnover_pct"] = round(total_turnover * 100, 2)
        out["total_cost_pct"] = round(total_cost * 100, 3)

    port_returns = curve.pct_change().dropna()
    if bench_returns is not None and len(bench_returns) > 0:
        b = pd.Series(bench_returns).astype(float)
        b.index = pd.to_datetime(b.index)
        aligned = pd.concat([port_returns.rename("p"), b.rename("b")], axis=1, sort=True).dropna()
        if len(aligned) > 1:
            bench_curve = (1 + aligned["b"]).cumprod()
            out["benchmark_curve"] = [
                {"date": d.strftime("%Y-%m-%d"), "nav": round(float(v), 4)}
                for d, v in bench_curve.items()
            ]
            out["benchmark_return_pct"] = round((bench_curve.iloc[-1] - 1) * 100, 2)
            out["excess_return_pct"] = round(
                stats.get("total_return_pct", 0) - out["benchmark_return_pct"], 2)
            beta, alpha = M.beta_alpha(aligned["p"], aligned["b"])
            out["beta"] = beta
            out["alpha_pct"] = alpha
            # 跟踪误差与信息比率：跑输/跑赢基准是不是有足够稳定性
            diff = aligned["p"] - aligned["b"]
            te = float(diff.std(ddof=1) * np.sqrt(trading_days)) if len(diff) > 1 else 0.0
            out["tracking_error_pct"] = round(te * 100, 2)
            out["information_ratio"] = round(float(diff.mean() * trading_days / te), 3) if te > EPS else 0.0
    return out
