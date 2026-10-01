"""仓位策略：删掉 K 线，用 Sigmoid 天平代替 if-else 长尾。

对应第三期（化繁为简、彻底删掉 K 线、Sigmoid 动态天平秒杀 1000 行 if-else）与
第四期（放弃精确预测、把绝对价格 P 降维成 Boolean、用 Sigmoid 约束高次疯狂）。

三层参数架构照搬 GA 设计文档的"环境—季节—基因"分层，并且**严格遵守一条红线**：

    Environment（物理法则）不参与搜索、不被任何优化改写；
    它写死了 long-only、杠杆上限 1、全局回撤熔断、费用下限。
    StrategyGenes（战斗基因）才是可以被 analysis.search 进化的部分。

这条边界很重要：一旦把"杠杆上限""是否允许做空"放进了可搜索的参数盒，
遗传算法一定会在某个窗口里找到"加杠杆更高分"的解——那不是策略，那是过拟合
加自杀。所以本项目在结构上就不给这个可能性。

仓位不是预测，是风险预算的连续映射：
    exposure = ceiling · Π sigmoid_gate(badness_i; threshold_i, gain_i)
三个状态量：已实现波动相对目标的超额、当前回撤、加速度 A（ABG 二阶导）。
gain 越大越接近硬阈值；gain 越小越平滑。默认取中等偏硬，
因为它必须先在样本内表现得"像个常识规则"，才谈得上被进化。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.abg import ABGParams, abg_filter
from analysis.constants import EPS, TRADING_DAYS


# ========== 第一层：环境（不可搜索） ==========
@dataclass(frozen=True)
class Environment:
    """
    造物主法则：物理阻断与底线红线。

    这些值在任何优化/搜索流程中都被视为常量。analysis.search 的参数盒
    只从 StrategyGenes 生成，不提供触碰这里字段的途径。
    """
    max_leverage: float = 1.0          # 现货基线 1.0：本项目永不加杠杆
    long_only: bool = True             # 公募基金无法做空
    min_exposure: float = 0.0          # 底仓下限（0 表示允许清仓观望）
    global_stop_loss: float = 0.25     # 全局权益回撤熔断，0 表示关闭
    one_way_fee: float = 0.003         # 单边费用下限（申购+赎回摊到单边）
    execution_lag_days: int = 1        # T 日信号 T+1 才可能确认：基金净值滞后

    def guard(self) -> Dict[str, object]:
        return {
            "max_leverage": self.max_leverage,
            "long_only": self.long_only,
            "global_stop_loss": self.global_stop_loss,
            "execution_lag_days": self.execution_lag_days,
            "notes": [
                "杠杆上限与做空权限是常量，不进入任何参数搜索空间",
                f"信号至少滞后 {self.execution_lag_days} 日执行，杜绝用当日净值决定当日申赎",
            ],
        }


DEFAULT_ENV = Environment()


# ========== 第二层：季节（跨 epoch 重采样，不逐日优化） ==========
SEASON_MULTIPLIERS: Dict[str, float] = {
    "冬": 0.55, "春": 0.80, "夏": 1.00, "秋": 0.90,
}
DEFAULT_SEASON = "夏"


def season_multiplier(season: Optional[str]) -> float:
    """资金激进乘数：季节层只影响"出多少钱"，不改变风险红线。"""
    return SEASON_MULTIPLIERS.get((season or DEFAULT_SEASON).strip(), 1.0)


# ========== 第三层：战斗基因（可进化） ==========
@dataclass
class StrategyGenes:
    """可被 GA 进化的参数。每个都有明确的物理含义与合法区间。"""
    # 波动率天平（逆波动率缩放，无陡峭度参数：vol_gain 已删除，因为它在
    # 改成比例缩放后不再影响任何输出——留下会让 GA 白耗探索预算）
    vol_target: float = 0.15           # 目标年化波动率
    vol_tolerance: float = 0.35        # 允许的超额倍数，超过才开始减仓
    # 回撤天平
    drawdown_mild: float = 0.10
    drawdown_severe: float = 0.22
    mild_hold: float = 0.65
    severe_hold: float = 0.30
    dd_gain: float = 12.0
    # 加速度天平（ABG 二阶导，归一化到波动率尺度）
    accel_gain: float = 10.0
    accel_threshold: float = -0.35     # A/σ 低于此值视为"加速下跌"
    # 位置天平（相对长期均线溢价）
    premium_threshold: float = 0.15
    premium_gain: float = 12.0
    # 摩擦控制：MinTradeThreshold，偏差小于此不调仓（过滤震荡市无效交易）
    min_trade_delta: float = 0.08
    rebalance_stride: int = 5          # 每 N 个交易日重新评估一次状态
    ma_window: int = 120
    vol_window: int = 63
    min_history: int = 30
    abg: ABGParams = field(default_factory=ABGParams)

    def clipped(self) -> "StrategyGenes":
        ma_window = int(np.clip(self.ma_window, 10, 504))
        vol_window = int(np.clip(self.vol_window, 10, 252))
        # 自洽：滚动窗口的 min_periods 不能超过窗口本身
        min_history = int(np.clip(self.min_history, 10, max(10, min(ma_window, vol_window))))
        return StrategyGenes(
            vol_target=float(np.clip(self.vol_target, 0.02, 0.60)),
            vol_tolerance=float(np.clip(self.vol_tolerance, 0.0, 3.0)),
            drawdown_mild=float(np.clip(self.drawdown_mild, 0.02, 0.60)),
            drawdown_severe=float(np.clip(self.drawdown_severe, 0.03, 0.90)),
            mild_hold=float(np.clip(self.mild_hold, 0.0, 1.0)),
            severe_hold=float(np.clip(self.severe_hold, 0.0, 1.0)),
            dd_gain=float(np.clip(self.dd_gain, 0.5, 40.0)),
            accel_gain=float(np.clip(self.accel_gain, 0.0, 20.0)),
            accel_threshold=float(np.clip(self.accel_threshold, -3.0, 0.0)),
            premium_threshold=float(np.clip(self.premium_threshold, 0.0, 1.0)),
            premium_gain=float(np.clip(self.premium_gain, 0.5, 40.0)),
            min_trade_delta=float(np.clip(self.min_trade_delta, 0.0, 0.60)),
            rebalance_stride=int(np.clip(self.rebalance_stride, 1, 63)),
            ma_window=ma_window,
            vol_window=vol_window,
            min_history=min_history,
            abg=self.abg.clipped(),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["abg"] = asdict(self.abg)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyGenes":
        """
        从档案/请求体里的 dict 重建基因。

        to_dict 会把 abg 序列化成嵌套 dict，所以不能用 `StrategyGenes(**data)`
        直接反序列化 —— 那样 abg 字段会是 dict，clipped() 立刻炸。
        """
        from dataclasses import fields as _fields
        from analysis.abg import ABGParams
        payload = dict(data or {})
        abg_raw = payload.pop("abg", None)
        allowed = {f.name for f in _fields(cls)} - {"abg"}
        kwargs = {k: v for k, v in payload.items() if k in allowed and v is not None}
        genes = cls(**kwargs)
        if isinstance(abg_raw, ABGParams):
            genes.abg = abg_raw
        elif isinstance(abg_raw, dict):
            abg_allowed = {f.name for f in _fields(ABGParams)}
            genes.abg = ABGParams(**{k: v for k, v in abg_raw.items()
                                     if k in abg_allowed and v is not None})
        return genes.clipped()


DEFAULT_GENES = StrategyGenes()


def _gate(badness: float, threshold: float, gain: float) -> float:
    """Sigmoid 天平：badness 越超过 threshold，输出越接近 0。支持标量与数组。"""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(gain * (np.asarray(badness, dtype=float) - threshold)))


VOL_FLOOR = 0.5      # 加速度天平最多把仓位砍一半
PREMIUM_FLOOR = 0.7  # 溢价天平最多砍掉三成


def gate_factors(vol, dd, accel, premium, g: StrategyGenes) -> Dict[str, object]:
    """
    四个天平的**唯一定义**。scalar（问答）与向量（模拟/进化）共用一份，
    避免"两处公式不一致"这种最难查的 bug。

    - 波动：经典逆波动率 min(1, target·(1+tol)/vol)。
      这里**故意不用 sigmoid**：阈值型曲线在波动翻倍时会直接归零，
      形成"一有风吹草动就清仓"的悬崖；比例型缩放是连续的，也不会加杠杆。
    - 回撤：两档 sigmoid 连乘，各自带 mild_hold / severe_hold 地板。
    - 加速度：sigmoid，地板 0.5（最多减半）。
    - 溢价：sigmoid，地板 0.7（最多砍三成）。
    """
    vol_arr = np.nan_to_num(np.asarray(vol, dtype=float), nan=0.0, posinf=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_vol = np.where(vol_arr > EPS,
                        np.minimum(1.0, g.vol_target * (1 + g.vol_tolerance) / np.maximum(vol_arr, EPS)),
                        1.0)

    # 缺失状态量按"中性"处理：dd=0、accel 极大正值、premium 极小负值，
    # 三者都会让对应天平输出 ≈1，即"不知道就不动手"（最悲观原则的另一种表达：
    # 不用猜测去触发减仓）。
    dd_arr = np.nan_to_num(np.abs(np.asarray(dd, dtype=float)), nan=0.0)
    f_dd = (np.clip(g.mild_hold + (1 - g.mild_hold) * _gate(dd_arr, g.drawdown_mild, g.dd_gain), 0, 1)
            * np.clip(g.severe_hold + (1 - g.severe_hold) * _gate(dd_arr, g.drawdown_severe, g.dd_gain), 0, 1))

    a_arr = np.nan_to_num(np.asarray(accel, dtype=float), nan=99.0, posinf=99.0, neginf=-99.0)
    f_accel = VOL_FLOOR + (1 - VOL_FLOOR) * _gate(-a_arr, -g.accel_threshold, g.accel_gain)

    p_arr = np.nan_to_num(np.asarray(premium, dtype=float), nan=-99.0, posinf=99.0, neginf=-99.0)
    f_prem = PREMIUM_FLOOR + (1 - PREMIUM_FLOOR) * _gate(p_arr, g.premium_threshold, g.premium_gain)

    return {"vol": f_vol, "drawdown": f_dd, "accel": f_accel, "premium": f_prem}


def read_state(nav: pd.Series, genes: StrategyGenes = DEFAULT_GENES) -> Dict[str, Optional[float]]:
    """
    从净值序列读取状态。只用截至当前的信息，样本不足时返回 None（最悲观原则：
    信息不够就不表态，而不是猜一个）。
    """
    g = genes.clipped()
    s = pd.Series(nav).astype(float).dropna()
    n = len(s)
    state: Dict[str, Optional[float]] = {"observations": int(n)}
    if n < g.min_history:
        state.update(realized_volatility=None, drawdown=None, premium_ratio=None, accel=None)
        return state

    r = s.tail(max(g.vol_window, 2)).pct_change().dropna()
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else None
    state["realized_volatility"] = round(vol, 4) if vol is not None else None

    peak = float(s.cummax().iloc[-1])
    state["drawdown"] = round(float(s.iloc[-1] / peak - 1.0), 4) if peak > EPS else None

    ma = float(s.tail(max(g.ma_window, 2)).mean())
    state["premium_ratio"] = round(float(s.iloc[-1] / ma - 1.0), 4) if ma > EPS else None

    filtered = abg_filter(s, params=g.abg)
    if filtered.empty or vol is None or vol <= EPS:
        state["accel"] = None
    else:
        # 归一化：A 的量纲是"净值/日²"，除以日波动尺度后才能跨标的比较
        state["accel"] = float(round(
            float(filtered["a"].iloc[-1]) / (vol / np.sqrt(TRADING_DAYS)), 4))
    return state


def exposure_from_state(state: Dict[str, Optional[float]],
                        genes: StrategyGenes = DEFAULT_GENES,
                        env: Environment = DEFAULT_ENV,
                        aggressiveness: float = 1.0) -> Dict[str, object]:
    """
    状态 -> 目标敞口。返回 exposure（占可用资金比例）+ 每个天平的输出与理由。

    四个天平连乘（公式唯一定义见 gate_factors，问答的标量路径与模拟/进化的
    向量路径共用）：波动逆比率缩放、回撤两档 sigmoid、加速度、均线溢价。
    状态良好时每个天平输出 ≈1，所以默认接近不动作；每个天平都有地板，
    任何单一状态恶化都不会直接把仓位砍到 0（避免"悬崖式清仓"）。
    """
    g = genes.clipped()
    if state.get("observations", 0) < g.min_history:
        return {"exposure": env.max_leverage, "gates": [], "insufficient_data": True,
                "reasons": [f"样本仅 {state.get('observations')} 天（<{g.min_history}），信息不足，按最悲观原则不做主动操作"]}

    vol = state.get("realized_volatility")
    dd = state.get("drawdown")
    accel = state.get("accel")
    premium = state.get("premium_ratio")
    factors = gate_factors(vol, dd, accel, premium, g)

    def scalar(v):
        return float(np.asarray(v, dtype=float))

    gates: List[Dict[str, object]] = []
    exposure = 1.0
    detail = [
        ("vol", factors["vol"],
         f"已实现波动 {(vol or 0) * 100:.1f}%，目标 {g.vol_target * 100:.0f}%"
         f"（超出容差 {g.vol_tolerance * 100:.0f}% 后按逆波动率减仓，不加杠杆）" if vol is not None else None),
        ("drawdown", factors["drawdown"],
         f"当前回撤 {abs(float(dd)) * 100:.1f}%（两档阈值 "
         f"{g.drawdown_mild * 100:.0f}%/{g.drawdown_severe * 100:.0f}%）" if dd is not None else None),
        ("accel", factors["accel"],
         f"归一化加速度 {float(accel):.2f}σ/日²，负值表示下跌在加速" if accel is not None else None),
        ("premium", factors["premium"],
         f"相对长期均线 {float(premium) * 100:+.1f}%" if premium is not None else None),
    ]
    for name, factor, reason in detail:
        f = scalar(factor)
        if not np.isfinite(f):
            continue
        exposure *= f
        if reason:
            gates.append({"name": name, "factor": round(f, 4), "reason": reason})

    exposure = exposure * float(aggressiveness)
    exposure = float(np.clip(exposure, env.min_exposure, env.max_leverage))
    reasons = [gt["reason"] for gt in gates] or ["各项状态均良好，维持满仓"]
    return {"exposure": round(exposure, 4), "gates": gates, "reasons": reasons,
            "insufficient_data": False,
            "boolean_regime": {"above_ma": bool((premium or 0) > 0),
                               "drawdown_active": bool((dd or 0) <= -g.drawdown_mild),
                               "vol_breach": bool((vol or 0) > g.vol_target * (1 + g.vol_tolerance))}}


def target_exposure(nav: pd.Series, genes: StrategyGenes = DEFAULT_GENES,
                    env: Environment = DEFAULT_ENV, season: Optional[str] = None
                    ) -> Dict[str, object]:
    """便捷入口：净值序列 -> 当前目标敞口与理由。"""
    g = genes.clipped()
    return exposure_from_state(read_state(nav, g), g, env, season_multiplier(season))
