"""模拟盘用例编排：账户、下单、结算、自动跟踪、复盘。

账本本身（paper/account.py）是纯逻辑；这里负责喂它真实数据并落盘。
净值一律走 FundRepository，所以自动继承限流、重试、熔断与缓存降级。
"""
from __future__ import annotations

from datetime import date as _date
from typing import Dict, List, Optional

import pandas as pd

from analysis.archive import StrategyArchive, StrategyRecord
from analysis.risk_rules import DEFAULT_EXIT
from analysis.signals import StrategyGenes
from config import Settings
from core.errors import InvalidInput, NotFound
from paper.account import BUY, SELL, PaperAccount
from paper.engine import decide, step as engine_step
from paper.store import PaperStore
from sources.repository import FundRepository


def _latest_nav(repo: FundRepository, codes: List[str], days: int = 1200
                ) -> Dict[str, float]:
    out = {}
    for code in codes:
        try:
            series = repo.nav_history(code, days=days)
        except Exception:
            continue
        if len(series):
            out[code] = float(series.iloc[-1])
    return out


def _nav_series(repo: FundRepository, codes: List[str], days: int = 1200
                ) -> Dict[str, pd.Series]:
    out = {}
    for code in codes:
        try:
            series = repo.nav_history(code, days=days)
        except Exception:
            series = pd.Series(dtype=float)
        if len(series) > 1:
            out[code] = series
    return out


def _watch_codes(account: PaperAccount) -> List[str]:
    codes = {l.code for l in account.lots} | {o.code for o in account.orders}
    return sorted(codes)


def open_account(store: PaperStore, repo: FundRepository, settings: Settings,
                 name: str, capital: float = 100000.0, codes: Optional[List[str]] = None,
                 strategy_id: Optional[str] = None, seed_buy: bool = False,
                 on_date: Optional[str] = None,
                 archive: Optional[StrategyArchive] = None) -> Dict[str, object]:
    """
    开一个模拟账户。绑定冠军策略时自动记下冻结日 = 今天，
    从此以后的净值才算样本外证据。
    """
    codes = [str(c).strip() for c in (codes or []) if str(c).strip()]
    for code in codes:
        if len(code) != 6 or not code.isdigit():
            raise InvalidInput(f"基金代码格式错误：{code}")
    if capital <= 0:
        raise InvalidInput("初始资金必须为正")

    today = (on_date or _date.today().isoformat())[:10]
    archive = archive or StrategyArchive()
    frozen_at = today
    record: Optional[StrategyRecord] = None
    if strategy_id:
        record = archive.get(strategy_id)
        if record is None:
            raise NotFound(f"找不到策略 {strategy_id}")
    champion = archive.champion()
    if record is None and champion is not None:
        record = champion

    account = PaperAccount.create(name=name or "模拟盘", initial_capital=capital,
                                  on_date=today, fees=settings.fees,
                                  strategy_id=(record.id if record else None),
                                  frozen_at=frozen_at)
    if codes:
        account.fund_types = {c: (settings.fees.default_type) for c in codes}

    submitted = []
    if seed_buy and codes:
        per = round(capital / len(codes), 2)
        for code in codes:
            order = account.submit_buy(code, per, today, reason="建仓（等权）")
            submitted.append({"code": code, "amount": per, "status": order.status,
                              "note": order.note})
    store.save(account)
    return {"account": account.to_dict(), "submitted": submitted,
            "frozen_at": frozen_at,
            "frozen_meaning": f"策略在 {frozen_at} 冻结，只有这之后公布的净值才算样本外证据",
            "bound_strategy": (record.to_dict() if record else None)}


def list_accounts(store: PaperStore) -> Dict[str, object]:
    accounts = store.list_accounts()
    return {"count": len(accounts),
            "accounts": [{"id": a.id, "name": a.name, "created_at": a.created_at,
                          "initial_capital": a.initial_capital,
                          "equity": (a.equity_history[-1]["equity"] if a.equity_history else a.cash),
                          "positions": len({l.code for l in a.lots}),
                          "pending": len([o for o in a.orders if o.status == "pending"]),
                          "frozen_at": a.frozen_at, "strategy_id": a.strategy_id}
                         for a in accounts]}


def account_detail(store: PaperStore, repo: FundRepository, account_id: str,
                   days: int = 1200) -> Dict[str, object]:
    account = store.get(account_id)
    if account is None:
        raise NotFound(f"找不到账户 {account_id}")
    codes = _watch_codes(account)
    nav_map = _latest_nav(repo, codes, days)
    today = _date.today().isoformat()
    stats = account.stats()
    return {
        "account": account.to_dict(),
        "positions": account.positions(today, nav_map),
        "stats": stats,
        "latest_nav": nav_map,
        "orders": [o for o in reversed(account.orders)],
        "disclaimer": "模拟盘不含真实成交冲击与额度限制，且基金申购常有限额；"
                      "结果只能用于检验规则本身，不代表真实可得收益。",
    }


def place_order(store: PaperStore, repo: FundRepository, account_id: str, code: str,
                side: str, amount: Optional[float] = None, shares: Optional[float] = None,
                reason: str = "", on_date: Optional[str] = None) -> Dict[str, object]:
    account = store.get(account_id)
    if account is None:
        raise NotFound(f"找不到账户 {account_id}")
    code = str(code).strip()
    if len(code) != 6 or not code.isdigit():
        raise InvalidInput("基金代码必须是 6 位数字")
    side = (side or "").lower()
    today = (on_date or _date.today().isoformat())[:10]

    if side == BUY:
        if not amount or amount <= 0:
            raise InvalidInput("申购需要正的 amount")
        order = account.submit_buy(code, float(amount), today, reason=reason)
    elif side == SELL:
        order = account.submit_sell(code, float(shares) if shares else None, today,
                                    reason=reason, all_positions=(shares is None))
    else:
        raise InvalidInput("side 只能是 buy 或 sell")

    store.save(account)
    return {"order": order.__dict__ | {}, "status": order.status, "note": order.note,
            "cash_after": round(account.cash, 2),
            "note_t1": "T 日提交按 T+1 之后公布的净值确认，当天无法知道成交价"}


def settle(store: PaperStore, repo: FundRepository, account_id: str,
           on_date: Optional[str] = None) -> Dict[str, object]:
    """用最新公布的净值结算待确认订单并盯市。"""
    account = store.get(account_id)
    if account is None:
        raise NotFound(f"找不到账户 {account_id}")
    codes = _watch_codes(account)
    nav_map = _latest_nav(repo, codes)
    day = (on_date or _date.today().isoformat())[:10]
    confirmed = account.settle(day, nav_map)
    store.save(account)
    return {"date": day, "confirmed": [o.__dict__ for o in confirmed],
            "stats": account.stats(), "latest_nav": nav_map}


def auto_run(store: PaperStore, repo: FundRepository, settings: Settings,
             account_id: str, genes: Optional[StrategyGenes] = None,
             on_date: Optional[str] = None, exit_cfg=DEFAULT_EXIT,
             archive: Optional[StrategyArchive] = None) -> Dict[str, object]:
    """
    让账户按冠军策略自己走一天：结算 → 判状态 → 下单。
    这就是"自动模拟"的部分；决策全部来自代码里的规则，LLM 不参与。
    """
    account = store.get(account_id)
    if account is None:
        raise NotFound(f"找不到账户 {account_id}")
    if genes is None:
        champ = (archive or StrategyArchive()).champion()
        genes = StrategyGenes.from_dict(champ.genes) if champ else StrategyGenes()

    codes = _watch_codes(account) or list(account.fund_types or [])
    nav_by_code = _nav_series(repo, codes)
    if not nav_by_code:
        raise InvalidInput("账户还没有可跟踪的基金净值，请先申购或指定标的")
    result = engine_step(account, (on_date or _date.today().isoformat())[:10],
                         nav_by_code, genes, exit_cfg=exit_cfg)
    store.save(account)
    result["genes_used"] = genes.to_dict() if hasattr(genes, "to_dict") else None
    return result


def preview_decision(store: PaperStore, repo: FundRepository, account_id: str,
                     genes: Optional[StrategyGenes] = None,
                     archive: Optional[StrategyArchive] = None) -> Dict[str, object]:
    """只算不动：看看今天规则会怎么决定仓位（下单前先看一眼）。"""
    account = store.get(account_id)
    if account is None:
        raise NotFound(f"找不到账户 {account_id}")
    if genes is None:
        champ = (archive or StrategyArchive()).champion()
        genes = StrategyGenes.from_dict(champ.genes) if champ else StrategyGenes()
    nav_by_code = _nav_series(repo, _watch_codes(account) or list(account.fund_types or []))
    if not nav_by_code:
        raise InvalidInput("没有可跟踪的净值数据")
    return decide(account, _date.today().isoformat(), nav_by_code, genes)


def delete_account(store: PaperStore, account_id: str) -> Dict[str, object]:
    if store.get(account_id) is None:
        raise NotFound(f"找不到账户 {account_id}")
    store.remove(account_id)
    return {"deleted": account_id}
