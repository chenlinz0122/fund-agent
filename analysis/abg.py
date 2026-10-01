"""ABG（Alpha-Beta-Gamma）滤波器：用泰勒展开把净值序列降次成 P / V / A。

来源：《量化交易 + 遗传算法》《放弃精确预测！降次求解》《止盈止损》三期的公开设计文档
（docs_exit.md「ABG 滤波器平仓原理」）。这里按文档 §3–§11 忠实实现：

    预测：P_pred = P + V·dt + ½A·dt²   V_pred = V + A·dt   A_pred = A
    残差：R = P_act - P_pred
    修正：P += αR   V += βR/dt   A += γR/dt²
    尺度：Var_R ← EWMA(R²)，半衰期 H；R_σ = sqrt(Var_R)
    置信：S_close = exp(-|R| / max(R_σ, ε))
    抛物底：V<0 且 A>0 时 P_bottom = P + V²/(2A)

对基金场景的两处**必须改动**（照搬会出事）：

1. 文档里的默认参数（α=0.60, β=0.20, γ=0.05, H=30, k_stop=5）是针对 **1 分钟 K 线**的。
   本项目的输入是**日频公募基金净值**，量纲完全不同：日频噪声尺度更大、跳变更慢。
   所以这里给出的默认值是按日频重新缩放的**起点**，不是"作者验证过的参数"，
   任何用于结论的参数都必须经 analysis.search 的样本外验证与
   analysis.validation.deflated_sharpe_ratio 的尝试次数校正。
2. 基金净值是**收盘后才公布**的，T 日的 P/V/A 在 T 日盘中不可得。
   因此所有下游用法都必须把信号滞后一天（见 simulation.py），
   否则会造出"用当天净值决定当天买入"的前视偏差。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.constants import EPS

# 日频重缩放后的起点参数（原始文档默认值见注释，来自 1 分钟级）
ALPHA_DEFAULT = 0.35      # 文档 0.60：日频噪声更大，位置跟得更保守
BETA_DEFAULT = 0.10       # 文档 0.20
GAMMA_DEFAULT = 0.02      # 文档 0.05
HALF_LIFE_DEFAULT = 20.0  # 文档 30 bar：日频 20 天约等于一个月记忆
K_STOP_DEFAULT = 4.0      # 文档 5.0：日频残差分布更厚，阈值略降
MIN_OBS_DEFAULT = 20      # 少于此样本量不产出状态（最悲观原则：信息不足则不表态）


@dataclass
class ABGParams:
    alpha: float = ALPHA_DEFAULT
    beta: float = BETA_DEFAULT
    gamma: float = GAMMA_DEFAULT
    dt: float = 1.0
    half_life: float = HALF_LIFE_DEFAULT
    k_stop: float = K_STOP_DEFAULT
    min_observations: int = MIN_OBS_DEFAULT

    def clipped(self) -> "ABGParams":
        return ABGParams(
            alpha=float(np.clip(self.alpha, 0.01, 0.99)),
            beta=float(np.clip(self.beta, 0.0, 0.60)),
            gamma=float(np.clip(self.gamma, 0.0, 0.30)),
            dt=float(max(self.dt, 1e-6)),
            half_life=float(max(self.half_life, 2.0)),
            k_stop=float(np.clip(self.k_stop, 1.0, 12.0)),
            min_observations=max(int(self.min_observations), 5),
        )


DEFAULT_ABG = ABGParams()


def abg_filter(values, params: ABGParams = DEFAULT_ABG,
               entry_value: Optional[float] = None) -> pd.DataFrame:
    """
    对一条序列（净值/价格）逐点运行 ABG 滤波器。

    返回 DataFrame：raw / p / v / a / residual / r_sigma / s_close /
    p_bottom / bottom_valid / rho / stop_ready。
    样本不足 min_observations 时返回空表而不是硬凑状态（最悲观原则）。
    """
    p = params.clipped()
    s = pd.Series(values).astype(float).dropna()
    if len(s) < p.min_observations:
        return pd.DataFrame(columns=["raw", "p", "v", "a", "residual", "r_sigma",
                                     "s_close", "p_bottom", "bottom_valid",
                                     "rho", "stop_ready"])

    lam = 1.0 - np.exp(-np.log(2.0) / p.half_life)
    dt = p.dt

    P = float(s.iloc[0])
    V = 0.0
    A = 0.0
    var_r = 0.0
    entry = float(entry_value) if entry_value is not None else P

    rows: List[dict] = []
    for value in s.values:
        actual = float(value)
        p_pred = P + V * dt + 0.5 * A * dt * dt
        v_pred = V + A * dt
        a_pred = A

        r = actual - p_pred
        var_r = (1 - lam) * var_r + lam * r * r
        r_sigma = max(float(np.sqrt(var_r)), 1e-12)

        P = p_pred + p.alpha * r
        V = v_pred + p.beta * r / dt
        A = a_pred + p.gamma * r / (dt * dt)

        s_close = float(np.exp(-abs(r) / r_sigma))
        bottom_valid = bool(V < 0 and A > EPS)
        p_bottom = P + (V * V) / (2 * A) if bottom_valid else np.nan
        rho = abs(r) / r_sigma

        rows.append({
            "raw": actual, "p": P, "v": V, "a": A,
            "residual": r, "r_sigma": r_sigma, "s_close": s_close,
            "p_bottom": p_bottom, "bottom_valid": bottom_valid,
            "rho": rho, "stop_ready": bool(rho > p.k_stop),
        })

    out = pd.DataFrame(rows, index=s.index)
    out["take_ready"] = take_profit_ready(out["p"], entry, out["v"], fee_rate(out))
    return out


def fee_rate(state: pd.DataFrame, one_way_fee: float = 0.003) -> float:
    """
    把"双边 taker 费率"概念换成基金的申赎费率。

    文档 §7 用 2.4·f 作为止盈门槛（双边 + 20% 缓冲）。基金的 f 是
    申购费 + 赎回费之和，量级远大于交易所 taker，所以止盈门槛天然更高。
    """
    return float(one_way_fee)


def take_profit_ready(P, entry_value: float, V, one_way_fee: float,
                      coverage: float = 2.4) -> pd.Series:
    """
    文档 §7：浮盈需覆盖 2.4 倍单边费用，且速度已非负（跌势企稳）。

    多头持仓场景下收益方向取反：(P - entry)/entry >= 2.4f。
    """
    entry = float(entry_value) if entry_value and entry_value > EPS else 1.0
    gain = (np.asarray(P, dtype=float) - entry) / entry
    return pd.Series((gain >= coverage * one_way_fee) & (np.asarray(V, dtype=float) >= 0.0),
                     index=None if not hasattr(P, "index") else P.index)


def exit_decision(state: Dict[str, object], params: ABGParams = DEFAULT_ABG,
                  entry_value: Optional[float] = None,
                  one_way_fee: float = 0.003) -> Dict[str, object]:
    """
    文档 §10 的决策顺序（取最新一日的状态）：止损优先，其次止盈/抛物底，否则持有。

    输入是 abg_filter(...).iloc[-1].to_dict() 形式。异常/缺字段时返回"持有"，
    因为出场系统本身的崩溃比一次误操作更危险（文档 §00:43 那个"睡一觉利润全吐"的坑）。
    """
    try:
        p = params.clipped()
        rho = float(state.get("rho", 0.0) or 0.0)
        stop_ready = bool(state.get("stop_ready", False)) or rho > p.k_stop
        take_ready = bool(state.get("take_ready", False))
        bottom_valid = bool(state.get("bottom_valid", False))
        s_close = float(state.get("s_close", 0.0) or 0.0)
        current = float(state.get("raw", 0.0) or 0.0)
        bottom = float(state.get("p_bottom", np.nan))

        if stop_ready:
            return {"action": "exit", "priority": 1,
                    "reason": f"残差 {rho:.1f} 倍超过 k_stop={p.k_stop}，视为未预期的跳变，全额赎回",
                    "reference_price": current}
        if take_ready or bottom_valid:
            if bottom_valid and np.isfinite(bottom) and 0 < bottom < current:
                ref = current - s_close * (current - bottom)
                how = "抛物底估计 + 置信度加权"
            else:
                ref = current * 0.999
                how = "抛物底无效，回退为内侧偏移"
            return {"action": "reduce", "priority": 2,
                    "reason": f"止盈就绪（{how}），参考赎回净值 {ref:.4f}",
                    "reference_price": float(ref)}
        return {"action": "hold", "priority": 3,
                "reason": f"残差 {rho:.1f} 在阈值内且未达止盈条件，继续持有",
                "reference_price": current}
    except Exception as e:  # 永远不因异常而误触发交易
        return {"action": "hold", "priority": 99,
                "reason": f"状态异常，按最悲观原则不动作：{type(e).__name__}",
                "reference_price": None}


def boolean_regime(state: Dict[str, object]) -> Dict[str, bool]:
    """
    文档 §11:00「把 P（绝对价格）降维成 Boolean 开关」。

    绝对价格没有跨标的可比性，也不该进模型；只保留方向性判断：
    在加速还是减速、在残差之上还是之下、滤波器是否信任当前价。
    """
    def flag(key, default=0.0, cmp=lambda x: x > 0):
        try:
            return bool(cmp(float(state.get(key, default) or default)))
        except (TypeError, ValueError):
            return False

    return {
        "accelerating": flag("a"),
        "decelerating": flag("a", cmp=lambda x: x < 0),
        "rising": flag("v"),
        "falling": flag("v", cmp=lambda x: x < 0),
        "convex_bottom": bool(state.get("bottom_valid", False)),
        "residual_shock": bool(state.get("stop_ready", False)),
        "filter_confident": float(state.get("s_close", 0.0) or 0.0) > 0.6,
    }
