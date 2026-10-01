"""出场（赎回）规则层。

对应第六期《量化做空的致命巨坑：怎么设置止盈止损》。该期分享文档给出了
ABG 滤波器的平仓决策顺序（止损 → 止盈/抛物底 → 持有），这里把它落到
**公募基金能真正执行的动作**上，并且把两处"基金与期货的本质差别"写实：

1. 没有限价单。基金只能按"未知净值"赎回：T 日提交，按 T 日收盘后才知道的
   净值成交，QDII 还要 T+2。所以文档里的"挂限价退出"在基金场景退化成
   "择时赎回 + 一日滑点"。任何回测若不施加 execution lag，就是在偷看答案。
2. 止盈门槛必须覆盖费用。文档 §7 用 2.4×单边费率作为"这趟值不值得跑"的门槛；
   基金单边费（申购+赎回摊算）通常在 0.1%~0.8% 之间，且持有不足 7 天有
   1.5% 惩罚性赎回费——门槛天然比场内高一个量级，"频繁择时赎回"在结构上就不划算。

出场系统自身的可靠性优先级高于出场逻辑（文档里那个"睡一觉利润全吐"的坑）：
本模块所有函数在任何异常下都返回 **不动作**，绝不因为算错状态而误赎。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.abg import ABGParams, abg_filter, exit_decision
from analysis.constants import EPS
from analysis.signals import DEFAULT_ENV, Environment, StrategyGenes

# 基金场景下的出场现实（写给用这个工具的人，不是写给回测）
FUND_EXIT_REALITY: List[Dict[str, str]] = [
    {"item": "无法挂限价", "effect": "赎回按 T 日未知净值成交，天然承担 1 日滑点（QDII 2 日以上）"},
    {"item": "赎回到账慢", "effect": "资金在途期间既不能享受反弹也不能规避下跌，止损的保护是有空窗的"},
    {"item": "惩罚性赎回费", "effect": "持有不足 7 天费率 1.5%，短期止损几乎必然把小亏换成确定亏损"},
    {"item": "止损后错过反弹", "effect": "净值序列有均值回复成分，止损再入场的成本要用样本外检验衡量，不能假设能回到原仓位"},
    {"item": "做空不可用", "effect": "多头持仓的「止损」只能减到现金，不能翻空，所以止损的收益全部来自避免下跌，不来自捕捉下跌"},
]


@dataclass
class ExitConfig:
    """出场规则参数。abg 之外的规则都作用在"策略自身净值"上。"""
    use_abg_stop: bool = True
    time_stop_days: int = 0            # 持有超过 N 天仍未跑赢基准则减仓，0 关闭
    time_stop_min_excess_pct: float = 0.0
    profit_take_min_excess_pct: float = 0.0  # 达标止盈减仓，0 关闭
    profit_take_keep_ratio: float = 0.70     # 止盈后保留的仓位比例
    min_extra_observations: int = 30

    def clipped(self) -> "ExitConfig":
        return ExitConfig(
            use_abg_stop=bool(self.use_abg_stop),
            time_stop_days=int(np.clip(self.time_stop_days, 0, 2520)),
            time_stop_min_excess_pct=float(np.clip(self.time_stop_min_excess_pct, -50, 50)),
            profit_take_min_excess_pct=float(np.clip(self.profit_take_min_excess_pct, 0, 200)),
            profit_take_keep_ratio=float(np.clip(self.profit_take_keep_ratio, 0, 1)),
            min_extra_observations=max(int(self.min_extra_observations), 10),
        )


DEFAULT_EXIT = ExitConfig()


def no_trade_band(current: float, target: float, min_delta: float) -> Tuple[float, bool]:
    """
    文档里的 MinTradeThreshold：目标与实际偏差不足阈值就不动手。

    这是"极低摩擦成本"最主要的手段——它不是因为怕麻烦，
    而是因为震荡市里每一次无效调仓都要付确定的费用、换回不确定的收益。
    返回 (实际执行的目标仓位, 是否真的调仓)。
    """
    delta = float(target) - float(current)
    if abs(delta) < max(float(min_delta), 0.0):
        return float(current), False
    return float(target), True


def evaluate_exit(nav: pd.Series, genes: StrategyGenes, env: Environment = DEFAULT_ENV,
                  cfg: ExitConfig = DEFAULT_EXIT, entry_nav: Optional[float] = None,
                  days_held: int = 0, benchmark_nav: Optional[pd.Series] = None,
                  current_exposure: float = 1.0) -> Dict[str, object]:
    """
    按优先级给出出场决定。

    顺序：ABG 残差止损 > 全局回撤熔断 > 达标止盈减仓 > 时间止损 > 持有。
    任何异常都返回 hold。
    """
    g = genes.clipped()
    e = cfg.clipped()
    s = pd.Series(nav).astype(float).dropna()
    base = {"action": "hold", "factor": 1.0, "priority": 9, "reasons": [], "source": None}
    if len(s) < e.min_extra_observations:
        return {**base, "reasons": [f"净值样本 {len(s)} 天不足，最悲观原则：不动作"]}

    entry = float(entry_nav) if entry_nav else float(s.iloc[0])
    peak = float(s.cummax().iloc[-1])
    dd = float(s.iloc[-1] / peak - 1.0) if peak > EPS else 0.0

    try:
        # 1) ABG 残差止损（文档 §8/§10：跳变时立即全额退出）
        if e.use_abg_stop:
            filtered = abg_filter(s, params=g.abg, entry_value=entry)
            if not filtered.empty:
                decision = exit_decision(filtered.iloc[-1].to_dict(), params=g.abg,
                                         entry_value=entry, one_way_fee=env.one_way_fee)
                if decision["action"] == "exit":
                    return {"action": "exit", "factor": 0.0, "priority": 1,
                            "reasons": [decision["reason"]], "source": "abg_stop",
                            "reference_nav": decision["reference_price"]}
                if decision["action"] == "reduce":
                    base = {**base, "reduce_candidate": decision["reason"], "priority": 3}

        # 2) 全局权益回撤熔断（Environment 层红线，不可被基因突破）
        if env.global_stop_loss and dd <= -env.global_stop_loss:
            return {"action": "exit", "factor": 0.0, "priority": 2,
                    "reasons": [f"策略净值自高点回撤 {dd * 100:.1f}% 触及全局熔断 "
                                f"{env.global_stop_loss * 100:.0f}%"], "source": "global_stop",
                    "reference_nav": float(s.iloc[-1])}

        excess = _excess_return(s, benchmark_nav)

        # 3) 达标止盈减仓（多头场景的"止盈"= 减仓，不是翻空）
        if e.profit_take_min_excess_pct > 0 and excess >= e.profit_take_min_excess_pct:
            return {"action": "reduce", "factor": e.profit_take_keep_ratio, "priority": 3,
                    "reasons": [f"超额收益 {excess:.1f}% 达到止盈线 "
                                f"{e.profit_take_min_excess_pct:.1f}%，减仓至 "
                                f"{e.profit_take_keep_ratio * 100:.0f}%（多头不能翻空，止盈只是收手）"],
                    "source": "take_profit", "reference_nav": float(s.iloc[-1])}

        # 4) 时间止损：拿久了还没跑赢基准，说明占用的是机会成本
        if e.time_stop_days and days_held >= e.time_stop_days and excess <= e.time_stop_min_excess_pct:
            factor = min(current_exposure, 0.5)
            return {"action": "reduce", "factor": factor, "priority": 4,
                    "reasons": [f"已持有 {days_held} 天，超额收益 {excess:.1f}% 仍未达标，"
                                f"按时间止损降仓（资金的机会成本也是成本）"],
                    "source": "time_stop", "reference_nav": float(s.iloc[-1])}

        return {**base, "reference_nav": float(s.iloc[-1]),
                "reasons": [f"回撤 {dd * 100:.1f}%、超额 {excess:.1f}%、持有 {days_held} 天，均未触发出场条件"]}
    except Exception as exc:
        return {**base, "reasons": [f"出场状态计算异常（{type(exc).__name__}），按最悲观原则不动作"]}


def _excess_return(nav: pd.Series, benchmark_nav: Optional[pd.Series]) -> float:
    """策略区间收益减基准区间收益（百分数）。基准缺失时返回策略自身收益。"""
    s = pd.Series(nav).astype(float).dropna()
    if len(s) < 2:
        return 0.0
    own = float(s.iloc[-1] / s.iloc[0] - 1.0) * 100
    if benchmark_nav is None:
        return round(own, 3)
    b = pd.Series(benchmark_nav).astype(float).dropna()
    if len(b) < 2:
        return round(own, 3)
    aligned = pd.concat([s.rename("s"), b.rename("b")], axis=1, sort=True).dropna()
    if len(aligned) < 2:
        return round(own, 3)
    bench = float(aligned["b"].iloc[-1] / aligned["b"].iloc[0] - 1.0) * 100
    return round(own - bench, 3)


def describe_rules(genes: StrategyGenes, env: Environment = DEFAULT_ENV,
                   cfg: ExitConfig = DEFAULT_EXIT) -> Dict[str, object]:
    """把当前生效的仓位与出场规则讲成人话（前端与问答都用这个）。"""
    g, e, c = genes.clipped(), env, cfg.clipped()
    return {
        "position_policy": {
            "vol_target_pct": round(g.vol_target * 100, 1),
            "vol_tolerance_pct": round(g.vol_tolerance * 100, 1),
            "drawdown_bands_pct": [round(g.drawdown_mild * 100, 1), round(g.drawdown_severe * 100, 1)],
            "no_trade_band_pct": round(g.min_trade_delta * 100, 1),
            "rebalance_stride": g.rebalance_stride,
            "leverage_cap": e.max_leverage,
            "long_only": e.long_only,
        },
        "exit_rules": [
            {"name": "ABG 残差止损", "on": c.use_abg_stop, "detail": f"|R|/Rσ > {g.abg.k_stop}"},
            {"name": "全局回撤熔断", "on": bool(e.global_stop_loss),
             "detail": f"策略净值回撤 ≥ {e.global_stop_loss * 100:.0f}% 全额退出"},
            {"name": "达标止盈减仓", "on": c.profit_take_min_excess_pct > 0,
             "detail": f"超额 ≥ {c.profit_take_min_excess_pct}% 时保留 {c.profit_take_keep_ratio * 100:.0f}%"},
            {"name": "时间止损", "on": c.time_stop_days > 0,
             "detail": f"持有 ≥ {c.time_stop_days} 天仍未跑赢基准则降仓"},
        ],
        "fund_exit_reality": FUND_EXIT_REALITY,
        "guard": env.guard(),
    }
