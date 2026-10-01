"""期望值与费用侵蚀分析 —— 视频里那条"残酷数学真相"的落点。

原系统的筛选逻辑只看"近 1 年收益"，这在数学期望上是危险的：

  1. 收益是毛的，成本是确定要付的。申购费、赎回费、管理费、托管费会
     每年、每次申赎真实地扣掉一块，把"毛正期望"变成"净负期望"。
  2. 持有不足 7 天有惩罚性赎回费（不低于 1.5%），频繁申赎是稳定亏损源。
  3. 做空在期望上为负（借券成本 + 上游风险溢价 + 保证金/强平），
     杠杆只放大方差、不改变期望 —— 公募Long-only 场景里体现为
     "不得出现负权重、总敞口不得超过 100%"。
  4. 高收益往往来自极少数交易日（运气/噪声），不是可持续的期望。

本模块只依赖 numpy/pandas 和配置里的费率假设，全部离线可算、可单测。
费率是**估计值**，真实费率以基金法律文件为准。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.constants import DEFAULT_RISK_FREE, EPS
from config import AnalysisConfig, FeeConfig


@dataclass(frozen=True)
class FeeProfile:
    """单只基金的费用假设（百分数）。"""
    subscription: float = 0.15     # 申购费（主流平台折后）
    redemption: float = 0.50       # 赎回费（正常持有期）
    management: float = 1.20       # 年管理费
    custody: float = 0.20          # 年托管费
    fund_type: str = "混合型"

    @property
    def annual_internal(self) -> float:
        """每年从净值里计提的费用，无法回避。"""
        return self.management + self.custody

    def to_dict(self) -> dict:
        return asdict(self)


def fee_profile(fund_type: Optional[str], cfg: Optional[FeeConfig] = None) -> FeeProfile:
    """按基金类型匹配费率模板；匹配不到用默认模板并标注。"""
    cfg = cfg or FeeConfig()
    normalized = _normalize_type(fund_type)
    profile = cfg.profiles.get(normalized) or cfg.profiles[cfg.default_type]
    return FeeProfile(
        subscription=float(profile.get("subscription", 0.0)),
        redemption=float(profile.get("redemption", 0.0)),
        management=float(profile.get("management", 0.0)),
        custody=float(profile.get("custody", 0.0)),
        fund_type=normalized or cfg.default_type,
    )


def _normalize_type(fund_type: Optional[str]) -> str:
    if not fund_type:
        return ""
    t = str(fund_type).strip()
    if t.endswith("型"):
        t = t[:-1] + "型"
    elif t in {"股票", "混合", "债券", "指数", "货币", "QDII"}:
        t = t + "型"
    return t


def fee_drag(profile: FeeProfile, holding_years: float = 1.0,
             round_trips_per_year: float = 1.0,
             penalty_days: int = 7, penalty_redemption: float = 1.5) -> Dict[str, object]:
    """
    估算费用对年化收益的侵蚀（百分数）。

    internal_cost_pct      : 管理费+托管费，按年计提
    transaction_cost_pct   : 持有期内的申赎费合计
    penalty_applies        : 平均持有天数 < 惩罚阈值时触发
    annualized_drag_pct    : 摊到每年、用于和年化收益直接相减的总拖累
    """
    holding_years = max(float(holding_years or 1.0), 1.0 / 365)
    trips = max(float(round_trips_per_year or 0.0), 0.0) * holding_years
    avg_hold_days = (holding_years * 365.0) / max(trips, EPS)

    per_trip = profile.subscription + profile.redemption
    if avg_hold_days < penalty_days:
        per_trip = profile.subscription + penalty_redemption

    transaction_total = per_trip * max(trips, 0.0)
    internal_total = profile.annual_internal * holding_years
    annualized = (transaction_total / holding_years) + profile.annual_internal

    return {
        "profile": profile.to_dict(),
        "holding_years": round(holding_years, 3),
        "round_trips_per_year": round(float(round_trips_per_year or 0.0), 2),
        "avg_holding_days": round(avg_hold_days, 1) if trips > 0 else None,
        "internal_cost_pct": round(internal_total, 3),
        "transaction_cost_pct": round(transaction_total, 3),
        "total_cost_pct": round(internal_total + transaction_total, 3),
        "annualized_drag_pct": round(annualized, 3),
        "penalty_applies": bool(avg_hold_days < penalty_days and trips > 0),
    }


def luck_concentration(returns: pd.Series, top_n: int = 5) -> Dict[str, object]:
    """
    收益集中度：前 top_n 个交易日贡献了正负收益净额的多少。

    比例越高，说明这段"优秀业绩"越依赖少数几天 —— 更接近噪声而不是期望。
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < top_n + 1:
        return {"available": False, "note": "样本太短，无法估计收益集中度"}
    total = float(r.sum())                       # 对数收益近似
    pos = float(r[r > 0].sum())
    top = float(r.nlargest(top_n).sum())
    share_of_positive = top / pos if pos > EPS else None
    share_of_net = top / total if abs(total) > EPS else None
    return {
        "available": True,
        "top_n": top_n,
        "top_days_contribution_pct": round(top * 100, 2),
        "share_of_gain_pct": round(share_of_positive * 100, 2) if share_of_positive is not None else None,
        "share_of_net_pct": round(share_of_net * 100, 2) if share_of_net is not None else None,
        "days_covered": round(top_n / len(r) * 100, 1),
    }


def negative_expectancy_checks(guard: Optional[dict] = None,
                               avg_holding_days: Optional[float] = None,
                               round_trips_per_year: Optional[float] = None,
                               data_points: Optional[int] = None,
                               ranked_by_return_only: bool = False,
                               luck: Optional[dict] = None,
                               cfg: Optional[AnalysisConfig] = None) -> List[Dict[str, object]]:
    """
    结构性负期望来源清单。每一项都说明"为什么这是期望为负的行为"。
    """
    cfg = cfg or AnalysisConfig()
    checks: List[Dict[str, object]] = []

    if guard:
        shorts = guard.get("short_weights") or []
        checks.append({
            "item": "做空",
            "triggered": bool(shorts),
            "why": "公募基金无法做空；做空需支付借券成本、并长期承担向上的风险溢价，"
                   "且亏损上不封顶，是期望为负的方向",
            "detail": f"负权重 {len(shorts)} 个" if shorts else "无负权重",
        })
        gross = float(guard.get("gross_exposure_pct", 0) or 0)
        checks.append({
            "item": "杠杆",
            "triggered": gross > cfg.max_gross_exposure * 100 + 1e-6,
            "why": "杠杆按比例放大收益与波动，但不改变期望；它让同等期望下"
                   "破产概率（先验触及强平线）显著上升",
            "detail": f"总敞口 {gross:.1f}%，上限 {cfg.max_gross_exposure * 100:.0f}%",
        })

    checks.append({
        "item": "频繁申赎",
        "triggered": bool(round_trips_per_year and round_trips_per_year > 2),
        "why": "每次申赎都支付确定费用，换手越高毛期望被吃掉越多；"
               "对公募而言交易成本是结构性负期望的主要来源",
        "detail": f"年往返 {round_trips_per_year:.1f} 次" if round_trips_per_year else "未设定",
    })
    checks.append({
        "item": "惩罚性赎回费",
        "triggered": bool(avg_holding_days and avg_holding_days < 7),
        "why": "持有不足 7 天赎回费不低于 1.5%，几乎必然覆盖掉短期波动的收益",
        "detail": f"平均持有 {avg_holding_days:.0f} 天" if avg_holding_days else "未设定",
    })
    checks.append({
        "item": "样本过短",
        "triggered": bool(data_points and data_points < cfg.min_sample_days),
        "why": "样本不足时统计量的噪声远大于信号，排名结果无法外推",
        "detail": f"{data_points} 个数据点" if data_points else "未提供",
    })

    if luck and luck.get("available"):
        share = luck.get("share_of_gain_pct")
        checks.append({
            "item": "把噪声当规律",
            "triggered": bool(share is not None and share > cfg.luck_concentration_warn * 100),
            "why": "收益主要来自极少数交易日时，它更接近一次运气而非可持续期望；"
                   "以此类基金做历史业绩外推会系统性高估未来",
            "detail": f"前 {luck.get('top_n')} 天贡献 {share}% 的正收益" if share is not None else "无法估计",
        })

    if ranked_by_return_only:
        checks.append({
            "item": "只看历史收益排名",
            "triggered": True,
            "why": "单一涨幅排名本质是追涨；基金业绩持续性在统计上很弱，"
                   "需要用样本外检验（walk-forward）而非收益率本身来支撑",
            "detail": "本次排序仅使用近 1 年收益",
        })

    return checks


def expectancy_report(metrics: Dict[str, object],
                      fund_type: Optional[str] = None,
                      returns: Optional[pd.Series] = None,
                      holding_years: Optional[float] = None,
                      round_trips_per_year: Optional[float] = None,
                      fee_cfg: Optional[FeeConfig] = None,
                      analysis_cfg: Optional[AnalysisConfig] = None,
                      guard: Optional[dict] = None,
                      ranked_by_return_only: bool = False) -> Dict[str, object]:
    """
    把"毛收益指标"翻译成"净期望判断"。

    输入 analysis.metrics.risk_metrics 的返回值（含 annual_return_pct / volatility_pct），
    输出费后年化收益、费后夏普、正负期望结论与告警清单。
    """
    fee_cfg = fee_cfg or FeeConfig()
    analysis_cfg = analysis_cfg or AnalysisConfig()
    risk_free_pct = analysis_cfg.risk_free * 100

    gross_annual = float(metrics.get("annual_return_pct", 0) or 0)
    vol = float(metrics.get("volatility_pct", 0) or 0)
    gross_sharpe = float(metrics.get("sharpe", 0) or 0)

    profile = fee_profile(fund_type, fee_cfg)
    drag = fee_drag(profile,
                    holding_years if holding_years is not None else fee_cfg.default_holding_years,
                    round_trips_per_year if round_trips_per_year is not None
                    else fee_cfg.default_round_trips_per_year,
                    penalty_days=fee_cfg.penalty_days,
                    penalty_redemption=fee_cfg.penalty_redemption)

    annualized_drag = float(drag["annualized_drag_pct"])
    net_annual = round(gross_annual - annualized_drag, 3)
    net_sharpe = round((net_annual - risk_free_pct) / vol, 3) if vol > EPS else 0.0

    luck = luck_concentration(returns) if returns is not None else {"available": False}
    checks = negative_expectancy_checks(
        guard=guard,
        avg_holding_days=drag.get("avg_holding_days"),
        round_trips_per_year=drag.get("round_trips_per_year"),
        data_points=metrics.get("data_points"),
        ranked_by_return_only=ranked_by_return_only,
        luck=luck if luck.get("available") else None,
        cfg=analysis_cfg,
    )

    triggered = [c["item"] for c in checks if c.get("triggered")]
    positive = net_annual > 0 and net_sharpe > 0

    if positive and not triggered:
        verdict = "费后仍为正期望，且未命中结构性负期望来源"
    elif positive:
        verdict = f"费后为正期望，但存在需要你确认的风险点：{'、'.join(triggered)}"
    elif gross_annual > 0 and net_annual <= 0:
        verdict = f"毛收益为正但费后为负：费用侵蚀 {annualized_drag}% 把期望拉成了负值"
    else:
        verdict = "费后期望非正，历史业绩不足以支撑继续持有"

    return {
        "gross_annual_return_pct": round(gross_annual, 2),
        "gross_sharpe": gross_sharpe,
        "annualized_fee_drag_pct": annualized_drag,
        "net_annual_return_pct": net_annual,
        "net_sharpe": net_sharpe,
        "positive_expectancy": bool(positive),
        "fee": drag,
        "luck_concentration": luck,
        "risk_of_ruin_notes": triggered,
        "checks": checks,
        "verdict": verdict,
        "assumption": "费率为主流互联网代销平台常见假设值，真实费率以基金法律文件为准",
    }
