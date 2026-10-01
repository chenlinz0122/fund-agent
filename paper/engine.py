"""模拟盘的自动交易：把"状态→仓位→出场"的规则变成一个会自己下单的账户。

这里没有任何预测：目标仓位来自 analysis.signals 的四个天平，
出场来自 analysis.risk_rules 的规则，下单只是把两者的差额翻译成
基金能执行的申赎动作，并且受不交易带（min_trade_delta）约束。

诚实性约束：
  - 决策只用**已公布**的净值（T 日只能看到 T 及以前的数据，成交在之后）；
  - 现金不足直接拒单，不透支；
  - 每次调仓都按费率实扣，不交易带内的偏差一律不动手（省下的都是真钱）。
"""
from __future__ import annotations

from datetime import date as _date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.metrics import to_returns
from analysis.risk_rules import (
    DEFAULT_EXIT, ExitConfig, evaluate_exit, no_trade_band,
)
from analysis.signals import (
    DEFAULT_ENV, Environment, StrategyGenes, exposure_from_state, read_state,
    season_multiplier,
)
from paper.account import BUY, SELL, PaperAccount


def portfolio_nav(account: PaperAccount, nav_by_code: Dict[str, pd.Series]) -> pd.Series:
    """
    账户所跟踪的"标的组合"净值：按当前持仓权重合成（无持仓时用等权）。

    状态量必须来自组合自身，而不是单只基金，否则天平读的是错的东西。
    """
    codes = sorted({l.code for l in account.lots} | set(nav_by_code))
    frames = {c: s for c, s in nav_by_code.items() if len(s) > 1}
    if not frames:
        return pd.Series(dtype=float)
    if not account.lots:
        base = pd.DataFrame({c: s / s.iloc[0] for c, s in frames.items()}).ffill().dropna(how="all")
        return base.mean(axis=1)
    weights = {}
    total = sum(l.shares for l in account.lots) or 1.0
    for lot in account.lots:
        weights[lot.code] = weights.get(lot.code, 0.0) + lot.shares / total
    aligned = pd.DataFrame({c: s / s.iloc[0] for c, s in frames.items()}).ffill().dropna(how="all")
    cols = [c for c in weights if c in aligned.columns]
    if not cols:
        return pd.Series(dtype=float)
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    return (aligned[cols].values @ w) / float((aligned[cols].values @ w)[0])


def decide(account: PaperAccount, on_date: str, nav_by_code: Dict[str, pd.Series],
           genes: StrategyGenes = StrategyGenes(), env: Environment = DEFAULT_ENV,
           exit_cfg: ExitConfig = DEFAULT_EXIT) -> Dict[str, object]:
    """给出今日的目标敞口、当前敞口，以及要提交的订单清单。"""
    g = genes.clipped()
    nav = portfolio_nav(account, nav_by_code)
    if len(nav) < g.min_history:
        return {"target_exposure": 1.0, "current_exposure": None, "orders": [],
                "reasons": [f"组合净值仅 {len(nav)} 天，不足以判断状态，按最悲观原则不下单"]}

    state = read_state(nav, g)
    stance = exposure_from_state(state, g, env, season_multiplier(None))
    target = float(stance["exposure"])

    equity_point = next((p for p in reversed(account.equity_history) if p["date"] <= on_date), None)
    equity = float(equity_point["equity"]) if equity_point else account.cash
    holdings = float(equity_point["holdings"]) if equity_point else 0.0
    current = holdings / equity if equity > 1e-9 else 0.0

    exit_call = evaluate_exit(nav, genes=g, env=env, cfg=exit_cfg,
                              entry_nav=float(equity_point["equity"]) if equity_point else None,
                              days_held=len([p for p in account.equity_history if p["date"] <= on_date]),
                              current_exposure=current)
    if exit_call["action"] == "exit":
        target = 0.0
    elif exit_call["action"] == "reduce":
        target = min(target, current * float(exit_call["factor"]))

    applied, changed = no_trade_band(current, target, g.min_trade_delta)
    orders: List[dict] = []
    if changed and equity > 0:
        delta_value = (applied - current) * equity
        if delta_value > 0:
            orders.append({"side": BUY, "amount": round(delta_value, 2),
                           "reason": "；".join(stance["reasons"][:2])})
        else:
            orders.append({"side": SELL, "ratio": round(min(-delta_value / max(holdings, 1e-9), 1.0), 4),
                           "reason": "；".join(stance["reasons"][:2])})

    return {"target_exposure": round(target, 4), "applied_exposure": round(applied, 4),
            "current_exposure": round(current, 4), "equity": round(equity, 2),
            "holdings": round(holdings, 2), "orders": orders,
            "state": state, "gates": stance["gates"],
            "exit_call": {"action": exit_call["action"], "reasons": exit_call["reasons"]}}


def step(account: PaperAccount, on_date: str, nav_by_code: Dict[str, pd.Series],
         genes: StrategyGenes = StrategyGenes(), env: Environment = DEFAULT_ENV,
         exit_cfg: ExitConfig = DEFAULT_EXIT,
         benchmark_nav: Optional[pd.Series] = None) -> Dict[str, object]:
    """
    走一天：先结算昨天的待确认订单，再按规则下今天的新单。

    顺序很重要 —— 真实世界也是"先确认、后决策"，反过来会凭空多出可用现金。
    """
    nav_map = {c: float(s.iloc[-1]) for c, s in nav_by_code.items() if len(s)}
    settled = account.settle(on_date, nav_map)
    plan = decide(account, on_date, nav_by_code, genes, env, exit_cfg)

    submitted = []
    for intent in plan["orders"]:
        if intent["side"] == BUY:
            # 分散到当前持仓标的；空仓时按可用代码等权建仓
            targets = sorted({l.code for l in account.lots}) or sorted(nav_map)
            if not targets:
                continue
            per = round(intent["amount"] / len(targets), 2)
            for code in targets:
                if per < 1:
                    break
                order = account.submit_buy(code, per, on_date, reason=intent["reason"])
                submitted.append({"id": order.id, "code": code, "side": BUY,
                                  "amount": per, "status": order.status, "note": order.note})
        else:
            ratio = intent["ratio"]
            for code in sorted({l.code for l in account.lots}):
                shares = round(account.shares_of(code) * ratio, 2)
                if shares <= 0:
                    continue
                order = account.submit_sell(code, shares, on_date, reason=intent["reason"])
                submitted.append({"id": order.id, "code": code, "side": SELL,
                                  "shares": shares, "status": order.status, "note": order.note})

    return {"date": on_date, "plan": plan, "settled": [
                {"id": o.id, "code": o.code, "side": o.side, "nav": o.nav,
                 "fee": o.fee, "confirmed_date": o.confirmed_date} for o in settled],
            "submitted": submitted, "equity": account.stats()}
