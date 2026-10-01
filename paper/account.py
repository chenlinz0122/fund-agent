"""基金模拟盘账本（纯逻辑，无 I/O，可单测）。

撮合规则刻意做得"不舒服"，因为真实的基金申赎就是不舒服的：

  1. T 日提交订单时**不知道成交价**，只能按 T+1 之后第一个公布净值确认；
  2. 申购按金额、赎回按份额，份额要等到确认才知道；
  3. 赎回费按每一笔份额批次的实际持有天数阶梯计收，<7 天是惩罚性费率；
  4. 确认之后不能撤销，只能再下一笔反向订单（又要付一次费用）。

账本用"份额批次(lot)"而不是平均成本记账，就是为了第 3 条能算准。
所有日期都用 ISO 字符串（YYYY-MM-DD），避免时区与序列化歧义。
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from typing import Dict, List, Optional

from config import FeeConfig
from analysis.constants import EPS

BUY = "buy"
SELL = "sell"
PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"


def _d(value: str) -> _date:
    return _date.fromisoformat(str(value)[:10])


def _days_between(later: str, earlier: str) -> int:
    return max((_d(later) - _d(earlier)).days, 0)


@dataclass
class Lot:
    """一笔申购确认后的份额批次。"""
    code: str
    opened_date: str
    shares: float
    cost: float                 # 含申购费的实付金额
    subscription_fee: float

    @property
    def unit_cost(self) -> float:
        return self.cost / self.shares if self.shares > EPS else 0.0

    def holding_days(self, on_date: str) -> int:
        return _days_between(on_date, self.opened_date)


@dataclass
class Order:
    id: str
    code: str
    side: str                   # buy | sell
    submitted_date: str
    amount: Optional[float] = None      # 申购金额（元）
    shares: Optional[float] = None      # 赎回份额；None + amount=None 表示"全部"
    status: str = PENDING
    reason: str = ""
    # 确认后才有的字段
    confirmed_date: Optional[str] = None
    nav: Optional[float] = None
    fee: Optional[float] = None
    filled_amount: Optional[float] = None
    filled_shares: Optional[float] = None
    note: str = ""


@dataclass
class PaperAccount:
    """一个模拟账户。所有方法都是确定性的，数据由调用方喂进来。"""
    id: str
    name: str
    initial_capital: float
    cash: float
    created_at: str
    lots: List[Lot] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    equity_history: List[Dict[str, float]] = field(default_factory=list)
    strategy_id: Optional[str] = None        # 关联的冠军/挑战者档案
    frozen_at: Optional[str] = None          # 策略冻结日：只有这之后的成交才算样本外
    fees_config: FeeConfig = field(default_factory=FeeConfig)
    fund_types: Dict[str, str] = field(default_factory=dict)
    lot_selection: str = "oldest"            # 赎回时先动哪个批次

    # ---------- 构造 ----------
    @classmethod
    def create(cls, name: str, initial_capital: float = 100000.0,
               on_date: Optional[str] = None, fees: Optional[FeeConfig] = None,
               strategy_id: Optional[str] = None, frozen_at: Optional[str] = None) -> "PaperAccount":
        today = (on_date or _date.today().isoformat())[:10]
        acct = cls(
            id="pa-" + uuid.uuid4().hex[:8], name=name, initial_capital=float(initial_capital),
            cash=float(initial_capital), created_at=today, fees_config=fees or FeeConfig(),
            strategy_id=strategy_id, frozen_at=frozen_at,
        )
        acct.equity_history.append({"date": today, "cash": acct.cash, "holdings": 0.0,
                                    "equity": acct.cash, "exposure_pct": 0.0})
        return acct

    # ---------- 下单 ----------
    def submit_buy(self, code: str, amount: float, on_date: str, reason: str = "") -> Order:
        code = str(code).strip()
        amount = round(float(amount), 2)
        order = Order(id=uuid.uuid4().hex[:10], code=code, side=BUY,
                      submitted_date=on_date[:10], amount=amount, reason=reason)
        if len(code) != 6 or not code.isdigit():
            order.status = REJECTED
            order.note = "基金代码必须是 6 位数字"
        elif amount <= 0:
            order.status = REJECTED
            order.note = "申购金额必须为正"
        elif amount > self.cash:
            order.status = REJECTED
            order.note = f"可用现金不足（{self.cash:.2f}）"
        else:
            self.cash -= amount          # 提交即锁定资金，因为之后无法撤单
        self.orders.append(order)
        return order

    def submit_sell(self, code: str, shares: Optional[float], on_date: str,
                    reason: str = "", all_positions: bool = False) -> Order:
        code = str(code).strip()
        held = self.shares_of(code)
        order = Order(id=uuid.uuid4().hex[:10], code=code, side=SELL,
                      submitted_date=on_date[:10], reason=reason)
        if all_positions or shares is None:
            order.shares = held
        else:
            order.shares = round(float(shares), 2)
        if order.shares <= 0:
            order.status = REJECTED
            order.note = "没有可赎回的份额"
        elif order.shares > held + 1e-6:
            order.status = REJECTED
            order.note = f"份额不足（持有 {held:.2f}）"
        else:
            order.shares = min(order.shares, held)
        self.orders.append(order)
        return order

    # ---------- 结算 ----------
    def settle(self, on_date: str, nav_map: Dict[str, float]) -> List[Order]:
        """
        用某日公布的净值结算所有待确认订单，并记录当日盯市结果。

        只确认"提交日之后"才有净值的订单（T+1 未知净值撮合）。
        返回本次被确认的订单列表。
        """
        on_date = on_date[:10]
        confirmed: List[Order] = []
        for order in self.orders:
            if order.status != PENDING:
                continue
            if _d(order.submitted_date) >= _d(on_date):
                continue                       # 当天提交，当天还不知道净值
            nav = float(nav_map.get(order.code, 0.0) or 0.0)
            if nav <= EPS:
                continue                       # 这只基金今天没有公布净值，继续挂着
            if order.side == BUY:
                self._confirm_buy(order, on_date, nav)
            else:
                self._confirm_sell(order, on_date, nav)
            confirmed.append(order)

        self._mark_to_market(on_date, nav_map)
        return confirmed

    def _confirm_buy(self, order: Order, on_date: str, nav: float) -> None:
        sub_rate = self.fees_config.subscription_rate(self.fund_types.get(order.code)) / 100.0
        net_amount = order.amount * (1 - sub_rate)
        shares = net_amount / nav
        order.status = CONFIRMED
        order.confirmed_date = on_date
        order.nav = round(nav, 4)
        order.fee = round(order.amount * sub_rate, 2)
        order.filled_shares = round(shares, 2)
        order.filled_amount = round(order.amount, 2)
        self.lots.append(Lot(code=order.code, opened_date=on_date, shares=shares,
                             cost=order.amount, subscription_fee=order.amount * sub_rate))

    def _confirm_sell(self, order: Order, on_date: str, nav: float) -> None:
        remaining = order.shares
        proceeds = 0.0
        fees = 0.0
        taken: List[Lot] = []
        # 批次选择：默认先赎最早的份额（持有久、费率低）。
        # 这不是"最优假设"而是默认会计顺序，真实交易里你指定不了批次。
        lots = sorted(self.lots, key=lambda l: l.opened_date,
                      reverse=(self.lot_selection == "newest"))
        for lot in lots:
            if lot.code != order.code or remaining <= 1e-8:
                continue
            take = min(lot.shares, remaining)
            days = lot.holding_days(on_date)
            rate = self.fees_config.redemption_rate(days) / 100.0
            gross = take * nav
            fee = gross * rate
            proceeds += gross - fee
            fees += fee
            remaining -= take
            lot.shares -= take
            taken.append(lot)

        self.lots = [l for l in self.lots if l.shares > 1e-8]
        order.status = CONFIRMED
        order.confirmed_date = on_date
        order.nav = round(nav, 4)
        order.fee = round(fees, 2)
        order.filled_shares = round(order.shares, 2)
        order.filled_amount = round(proceeds, 2)
        order.note = f"赎回来自 {len(taken)} 个批次"
        self.cash += proceeds

    def _mark_to_market(self, on_date: str, nav_map: Dict[str, float]) -> None:
        holdings = 0.0
        for lot in self.lots:
            nav = float(nav_map.get(lot.code, 0.0) or 0.0)
            holdings += lot.shares * (nav if nav > EPS else lot.unit_cost)
        pending_buy = sum(o.amount or 0 for o in self.orders
                          if o.status == PENDING and o.side == BUY)
        equity = self.cash + pending_buy + holdings
        if self.equity_history and self.equity_history[-1]["date"] == on_date:
            self.equity_history[-1] = {"date": on_date, "cash": round(self.cash, 2),
                                       "holdings": round(holdings, 2),
                                       "equity": round(equity, 2),
                                       "exposure_pct": round(holdings / equity * 100, 1) if equity > EPS else 0.0}
        else:
            self.equity_history.append({"date": on_date, "cash": round(self.cash, 2),
                                        "holdings": round(holdings, 2),
                                        "equity": round(equity, 2),
                                        "exposure_pct": round(holdings / equity * 100, 1) if equity > EPS else 0.0})

    # ---------- 查询 ----------
    def shares_of(self, code: str) -> float:
        return round(sum(l.shares for l in self.lots if l.code == code), 2)

    def positions(self, on_date: str, nav_map: Dict[str, float]) -> List[Dict[str, object]]:
        """当前持仓，按批次展开：持有天数与赎回费率一目了然。"""
        out = []
        for lot in sorted(self.lots, key=lambda l: (l.code, l.opened_date)):
            nav = float(nav_map.get(lot.code, 0.0) or 0.0) or lot.unit_cost
            days = lot.holding_days(on_date)
            rate = self.fees_config.redemption_rate(days)
            value = lot.shares * nav
            out.append({
                "code": lot.code, "opened_date": lot.opened_date,
                "shares": round(lot.shares, 2), "unit_cost": round(lot.unit_cost, 4),
                "nav": round(nav, 4), "market_value": round(value, 2),
                "unrealized_pnl": round(value - lot.cost, 2),
                "pnl_pct": round((value / lot.cost - 1) * 100, 2) if lot.cost > EPS else 0.0,
                "holding_days": days, "redemption_rate_pct": rate,
                "penalty_applies": days < self.fees_config.penalty_days,
                "liquidation_value": round(value * (1 - rate / 100), 2),
            })
        return out

    def out_of_sample_stats(self, nav_map: Optional[Dict[str, float]] = None) -> Dict[str, object]:
        """
        样本外纪律的量化表达：策略冻结日之后到底观察了多少天、成交了多少笔。

        没有冻结日或冻结日之后没有数据时，明确标记 evidence=False ——
        这时任何"表现很好"的说法都只是样本内成绩。
        """
        if not self.frozen_at:
            return {"frozen_at": None, "evidence": False, "days_observed": 0,
                    "note": "账户未绑定策略冻结日，无法区分样本内外"}
        after = [p for p in self.equity_history if p["date"] > self.frozen_at]
        trades = [o for o in self.orders if o.status == CONFIRMED and (o.confirmed_date or "") > self.frozen_at]
        if len(after) < 2:
            return {"frozen_at": self.frozen_at, "evidence": False, "days_observed": len(after),
                    "confirmed_trades": len(trades),
                    "note": "冻结日之后还没有新的净值数据，当前成绩全部来自样本内，不能作为验证依据"}
        start = float(after[0]["equity"])
        end = float(after[-1]["equity"])
        peak, mdd = -1e18, 0.0
        for point in after:
            equity = float(point["equity"])
            peak = max(peak, equity)
            if peak > EPS:
                mdd = min(mdd, equity / peak - 1.0)
        return {
            "frozen_at": self.frozen_at, "evidence": True,
            "days_observed": len(after), "confirmed_trades": len(trades),
            "return_pct": round((end / start - 1) * 100, 2) if start > EPS else 0.0,
            "max_drawdown_pct": round(mdd * 100, 2),
            "note": f"冻结日 {self.frozen_at} 之后累计观察 {len(after)} 个交易日",
        }

    def stats(self) -> Dict[str, object]:
        from analysis.metrics import risk_metrics
        series = {p["date"]: float(p["equity"]) for p in self.equity_history}
        curve = None
        if len(series) >= 2:
            import pandas as pd
            curve = pd.Series(series, dtype=float).sort_index()
        metrics = risk_metrics(curve / curve.iloc[0]) if curve is not None else {"error": "净值点不足"}
        fees_paid = round(sum(float(o.fee or 0) for o in self.orders if o.status == CONFIRMED), 2)
        confirmed = [o for o in self.orders if o.status == CONFIRMED]
        return {
            "initial_capital": round(self.initial_capital, 2),
            "cash": round(self.cash, 2),
            "equity": round(self.equity_history[-1]["equity"], 2) if self.equity_history else round(self.cash, 2),
            "positions_value": round(self.equity_history[-1]["holdings"], 2) if self.equity_history else 0.0,
            "pending_orders": len([o for o in self.orders if o.status == PENDING]),
            "rejected_orders": len([o for o in self.orders if o.status == REJECTED]),
            "confirmed_trades": len(confirmed),
            "fees_paid": fees_paid,
            "fee_drag_pct": round(fees_paid / max(self.initial_capital, EPS) * 100, 3),
            "metrics": metrics,
            "out_of_sample": self.out_of_sample_stats(),
        }

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "initial_capital": self.initial_capital,
            "cash": self.cash, "created_at": self.created_at,
            "strategy_id": self.strategy_id, "frozen_at": self.frozen_at,
            "lot_selection": self.lot_selection,
            "fund_types": self.fund_types,
            "lots": [asdict(l) for l in self.lots],
            "orders": [asdict(o) for o in self.orders],
            "equity_history": self.equity_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperAccount":
        acct = cls(
            id=data["id"], name=data.get("name", ""), initial_capital=float(data["initial_capital"]),
            cash=float(data["cash"]), created_at=data.get("created_at", ""),
            strategy_id=data.get("strategy_id"), frozen_at=data.get("frozen_at"),
            lot_selection=data.get("lot_selection", "oldest"),
            fund_types=data.get("fund_types", {}) or {},
            equity_history=data.get("equity_history", []) or [],
        )
        acct.lots = [Lot(**l) for l in data.get("lots", [])]
        acct.orders = [Order(**o) for o in data.get("orders", [])]
        return acct
