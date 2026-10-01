"""可插拔模拟引擎：把 状态→仓位→出场→成本 串成一条快速回测管线。

对应第二期说的"快速回测系统架构"：引擎本身不认识任何具体策略，
策略只是一组参数（StrategyGenes）加一层风险规则（ExitConfig）。
为了跑得够快（GA 要评估成千上万次），做了两处结构性设计：

  1. **状态量取自"标的组合自身净值"，一次算完整条序列**（波动率、回撤、
     溢价、ABG 的 P/V/A 全部向量化），而不是每天重算历史。
  2. **只有仓位与成本是逐日推进的**，因为它们有路径依赖（不交易带、
     熔断、费用）。

诚实性约束（写死在代码里，不靠自觉）：
  - t 日的敞口只能用 t-1 日及以前的状态决定（`execution_lag_days`，默认 1）。
    基金净值是收盘后公布的，任何"用当天净值决定当天申赎"都是前视偏差。
  - 敞口上限来自 Environment.max_leverage（默认 1.0），且这个值在本模块内
    不可被参数覆盖 —— 进化搜索碰不到它。
  - 换手一律计费，包括"止损后再入场"的那一次买入。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis import metrics as M
from analysis.abg import abg_filter
from analysis.constants import DEFAULT_RISK_FREE, EPS, TRADING_DAYS
from analysis.risk_rules import DEFAULT_EXIT, ExitConfig, evaluate_exit, no_trade_band
from analysis.signals import (
    DEFAULT_ENV, Environment, StrategyGenes, gate_factors, season_multiplier,
)


@dataclass
class SimulationResult:
    curve: pd.Series
    exposure: pd.Series
    trades: List[dict]
    costs_pct: float
    turnover_pct: float
    metrics: Dict[str, object]
    baseline_dca_pct: float
    benchmark_pct: Optional[float]

    def to_dict(self) -> dict:
        return {
            "equity_curve": [{"date": d.strftime("%Y-%m-%d"), "nav": round(float(v), 4)}
                             for d, v in self.curve.items()],
            "exposure_curve": [{"date": d.strftime("%Y-%m-%d"), "exposure": round(float(v) * 100, 1)}
                               for d, v in self.exposure.items()],
            "trades": self.trades,
            "trade_count": len(self.trades),
            "total_turnover_pct": round(self.turnover_pct, 2),
            "total_cost_pct": round(self.costs_pct, 3),
            "metrics": self.metrics,
            "baseline_dca_pct": self.baseline_dca_pct,
            "benchmark_return_pct": self.benchmark_pct,
        }


def _gates_frame(nav: pd.Series, genes: StrategyGenes) -> pd.DataFrame:
    """把四个天平一次性算成 DataFrame（列：vol/drawdown/accel/premium/exposure）。

    公式只有一份，来自 analysis.signals.gate_factors —— 问答里看到的理由
    和回测里用的仓位必须是同一个函数算出来的。
    """
    g = genes.clipped()
    returns = nav.pct_change()
    vol_win = max(g.vol_window, 5)
    ma_win = max(g.ma_window, 5)
    vol = returns.rolling(vol_win, min_periods=min(g.min_history, vol_win)).std(ddof=1) * np.sqrt(TRADING_DAYS)
    dd = (nav / nav.cummax() - 1.0).abs()
    ma = nav.rolling(ma_win, min_periods=min(g.min_history, ma_win)).mean()
    premium = nav / ma - 1.0

    accel = pd.Series(np.nan, index=nav.index)
    filtered = abg_filter(nav, params=g.abg)
    if not filtered.empty:
        daily_sigma = (vol.reindex(filtered.index) / np.sqrt(TRADING_DAYS)).replace(0, np.nan)
        accel.loc[filtered.index] = (filtered["a"].values / daily_sigma.values)

    fac = gate_factors(vol.values, dd.values, accel.values, premium.values, g)
    out = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in fac.items()}, index=nav.index)
    out["exposure"] = np.clip(out.prod(axis=1), 0.0, 1.0)
    return out


def dca_baseline(nav: pd.Series, months: int = 1) -> float:
    """
    Ghost DCA 基准：无脑定期定投的区间收益率（百分数）。

    文档把"跑赢无脑定投"当作唯一的 Alpha 定义。基金场景里这就是基准：
    任何策略如果连按月定投同一只基金都跑不赢，就没有继续讨论的价值。
    """
    s = pd.Series(nav).astype(float).dropna()
    if len(s) < 30:
        return 0.0
    step = max(int(months) * 21, 21)
    buys = s.iloc[::step]
    if len(buys) < 2:
        return 0.0
    units = (1.0 / buys).sum()
    invested = float(len(buys))
    return round(float((units * s.iloc[-1]) / invested - 1.0) * 100, 2)


def simulate(returns_df: pd.DataFrame, genes: Optional[StrategyGenes] = None,
             env: Environment = DEFAULT_ENV, exit_cfg: Optional[ExitConfig] = None,
             weights: Optional[Dict[str, float]] = None,
             benchmark_returns: Optional[pd.Series] = None,
             season: Optional[str] = None,
             risk_free: float = DEFAULT_RISK_FREE) -> Dict[str, object]:
    """
    运行一次模拟。返回 {result, policy, warnings}；数据不足时 result 为 None。
    """
    g = (genes or StrategyGenes()).clipped()
    cfg = (exit_cfg or DEFAULT_EXIT).clipped()
    df = returns_df.dropna(axis=1, how="all").dropna()
    cols = list(df.columns)
    if not cols or len(df) < g.min_history + 30:
        return {"error": f"对齐样本不足（{len(df)} 天，需要至少 {g.min_history + 30} 天）"}

    if weights:
        w = np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > EPS else np.ones(len(cols)) / len(cols)
    else:
        w = np.ones(len(cols)) / len(cols)

    # 标的组合自身净值（不带策略）：状态量的唯一来源
    base_ret = df.values @ w
    base_nav = pd.Series(np.cumprod(1 + base_ret), index=df.index)
    gates = _gates_frame(base_nav, g)
    target_path = gates["exposure"] * season_multiplier(season)

    rf_daily = risk_free / TRADING_DAYS
    exposure = float(env.max_leverage)          # 初始满仓（后续按天平调整）
    equity = [1.0]
    exposures: List[float] = []
    trades: List[dict] = []
    costs_total = 0.0
    turnover_total = 0.0
    entry_equity = 1.0
    days_held = 0
    since_check = 0
    equity_series = pd.Series(dtype=float)
    fused = False

    for i in range(len(base_ret)):
        # 关键：用 t-1 及以前的信息决定 t 日敞口
        hist = pd.Series(equity, index=df.index[:len(equity)])

        decided = exposure
        if not fused and i >= g.min_history:
            since_check += 1
            if since_check >= g.rebalance_stride:
                since_check = 0
                target = float(target_path.iloc[i - 1] if i >= 1 else target_path.iloc[0])
                new_exposure, changed = no_trade_band(decided, target, g.min_trade_delta)
                if changed:
                    decided = new_exposure
                    trades.append({"date": df.index[i].strftime("%Y-%m-%d"), "action": "rebalance",
                                   "from": round(exposure * 100, 1), "to": round(new_exposure * 100, 1),
                                   "target": round(target * 100, 1)})

            # 出场层：作用在策略自身净值上
            if not fused and len(equity) > cfg.min_extra_observations:
                decision = evaluate_exit(
                    hist, genes=g, env=env, cfg=cfg, entry_nav=entry_equity,
                    days_held=days_held,
                    benchmark_nav=(pd.Series(np.cumprod(1 + benchmark_returns.reindex(df.index[:i]).fillna(0).values),
                                             index=df.index[:i]) if benchmark_returns is not None and i > 0 else None),
                    current_exposure=decided,
                )
                if decision["action"] == "exit":
                    decided = float(env.min_exposure)
                    if decision.get("source") == "global_stop":
                        fused = True
                    trades.append({"date": df.index[i].strftime("%Y-%m-%d"), "action": "exit",
                                   "from": round(exposure * 100, 1), "to": round(decided * 100, 1),
                                   "reason": decision["reasons"][0] if decision["reasons"] else ""})
                elif decision["action"] == "reduce":
                    decided = float(np.clip(decided * float(decision["factor"]),
                                            env.min_exposure, env.max_leverage))
                    trades.append({"date": df.index[i].strftime("%Y-%m-%d"), "action": "reduce",
                                   "from": round(exposure * 100, 1), "to": round(decided * 100, 1),
                                   "reason": decision["reasons"][0] if decision["reasons"] else ""})

        decided = float(np.clip(decided, env.min_exposure, env.max_leverage))
        delta = abs(decided - exposure)
        if delta > 1e-9:
            fee = delta * env.one_way_fee
            costs_total += fee
            turnover_total += delta
            equity[-1] *= (1.0 - fee)     # 费用直接从权益里扣，不靠事后修正
        exposure = decided
        exposures.append(exposure)

        day_ret = exposure * float(base_ret[i]) + (1 - exposure) * rf_daily
        equity.append(equity[-1] * (1 + day_ret))
        days_held += 1

    curve = pd.Series(equity[1:], index=df.index, dtype=float)
    curve = curve / curve.iloc[0]
    exposures_s = pd.Series(exposures, index=df.index, dtype=float)

    stats = M.risk_metrics(curve, risk_free=risk_free)
    gross_cost_drag = round(costs_total * 100, 3)
    dca = dca_baseline(base_nav)

    out: Dict[str, object] = {
        "result": {
            "window": [df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")],
            "sample_days": int(len(df)),
            "funds": cols,
            "weights": {c: round(float(w[i] * 100), 2) for i, c in enumerate(cols)},
            "season": season or "夏",
            "aggressiveness": season_multiplier(season),
            "policy_used": {"vol_target_pct": round(g.vol_target * 100, 1),
                            "no_trade_band_pct": round(g.min_trade_delta * 100, 1),
                            "rebalance_stride": g.rebalance_stride,
                            "leverage_cap": env.max_leverage,
                            "execution_lag_days": env.execution_lag_days},
            "avg_exposure_pct": round(float(exposures_s.mean()) * 100, 1),
            "exposure_range_pct": [round(float(exposures_s.min()) * 100, 1),
                                   round(float(exposures_s.max()) * 100, 1)],
            "strategy": SimulationResult(
                curve=curve, exposure=exposures_s, trades=trades,
                costs_pct=gross_cost_drag, turnover_pct=turnover_total * 100,
                metrics=stats, baseline_dca_pct=dca,
                benchmark_pct=(round(float(np.prod(1 + np.asarray(benchmark_returns.dropna()))) * 100 - 100, 2)
                               if benchmark_returns is not None and len(benchmark_returns) else None),
            ).to_dict(),
            "alpha_vs_dca_pct": round(stats.get("total_return_pct", 0) - dca, 2),
            "guard": env.guard(),
            "fused": bool(fused),
        },
        "warnings": [],
    }
    if fused:
        out["warnings"].append("策略净值曾触及全局回撤熔断并清仓，之后的上涨不再属于该策略——这是设计意图，不是 bug")
    return out
