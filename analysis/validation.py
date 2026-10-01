"""样本外有效性检验与抗过拟合。

"回测猛如虎"通常是因为把噪声当成了规律。这个模块回答两个问题：

  A. 我的打分/排名对未来有没有预测力？—— 用 walk-forward：
     只用 T 之前的数据打分，再看 T 之后一段的真实表现，算秩相关（IC）。
     样本内排名好看而样本外 IC≈0，说明筛选器只是在复述历史。

  B. 这个夏普是不是"试出来的"？—— 用 Deflated Sharpe Ratio：
     在尝试了 N 个参数/策略之后，最好的那个 Sharpe 有相当大的概率纯属运气，
     尝试次数越多，门槛越高。DSR < 0.95 通常意味着不能拒绝"真实期望为 0"。

全部基于 numpy/pandas + 标准库 statistics，不需要 scipy，离线可跑。
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.constants import EPS, TRADING_DAYS
from analysis.metrics import annualized_return, to_returns

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649


def sharpe_summary(returns: pd.Series, risk_free_annual: float = 0.0,
                   trading_days: int = TRADING_DAYS) -> Dict[str, object]:
    """日频夏普及其抽样统计量（偏度、峰度、样本数、标准误）。"""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"available": False, "note": "样本不足 3 个交易日"}
    rf_daily = risk_free_annual / trading_days
    ex = r - rf_daily
    std = float(ex.std(ddof=1))
    sr_daily = float(ex.mean() / std) if std > EPS else 0.0
    sr_annual = sr_daily * float(np.sqrt(trading_days))
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0          # 转换为非中心峰度（正态=3）
    # Lo (2002) 修正标准误，考虑厚尾与偏度
    se = float(np.sqrt((1 - skew * sr_daily + ((kurt - 1) / 4.0) * sr_daily ** 2) / max(n - 1, 1)))
    return {
        "available": True,
        "observations": n,
        "sharpe_daily": round(sr_daily, 4),
        "sharpe_annualized": round(sr_annual, 3),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "std_error": round(se, 4),
    }


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """
    在 N 次独立尝试下，仅凭运气也能得到的"最好夏普"的期望（Bailey & López de Prado）。

    N 越大门槛越高 —— 这就是"参数搜索出来的漂亮回测"不该被相信的定量理由。
    """
    if n_trials <= 1:
        return 0.0
    z = _NORMAL.inv_cdf
    term = (1 - _EULER_GAMMA) * z(1 - 1.0 / n_trials) + _EULER_GAMMA * z(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(max(var_sr, 0.0)) * term)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int = 1,
                          risk_free_annual: float = 0.0,
                          var_sr: Optional[float] = None,
                          trading_days: int = TRADING_DAYS) -> Dict[str, object]:
    """
    DSR = P(真实夏普 > 0 | 你在 n_trials 个变体里挑了最好的那个)。

    var_sr 缺省时用该样本自身的抽样方差近似（保守做法）。
    """
    s = sharpe_summary(returns, risk_free_annual, trading_days)
    if not s.get("available"):
        return {"available": False, **s}

    sr = float(s["sharpe_daily"])
    t_obs = int(s["observations"])
    skew = float(s["skewness"])
    kurt = float(s["kurtosis"])
    se = float(s["std_error"])

    v = float(var_sr) if var_sr is not None and var_sr > EPS else se ** 2
    sr_star = expected_max_sharpe(max(int(n_trials), 1), v)

    denom = np.sqrt(max(1 - skew * sr + ((kurt - 1) / 4.0) * sr ** 2, EPS))
    z_stat = (sr - sr_star) * np.sqrt(max(t_obs - 1, 1)) / denom
    dsr = float(_NORMAL.cdf(z_stat))

    return {
        "available": True,
        "deflated_sharpe": round(dsr, 4),
        "observed_sharpe_annualized": s["sharpe_annualized"],
        "threshold_sharpe_daily": round(sr_star, 4),
        "threshold_sharpe_annualized": round(sr_star * float(np.sqrt(trading_days)), 3),
        "n_trials": int(n_trials),
        "observations": t_obs,
        "robust_at_95": bool(dsr >= 0.95),
        "interpretation": (
            "考虑尝试次数后，仍有 95% 以上的把握真实夏普为正" if dsr >= 0.95 else
            f"DSR={dsr:.2f}：不能拒绝「真实期望为 0」，很可能是搜索出来的噪声"
        ),
    }


def information_coefficient(scores: Dict[str, float],
                            forward_returns: Dict[str, float]) -> Dict[str, object]:
    """
    截面 IC：打分排名与未来收益排名的秩相关（Spearman）。

    IC≈0 表示打分对未来没有预测力，无论它在样本内看起来多准。
    """
    common = [c for c in scores if c in forward_returns]
    if len(common) < 3:
        return {"available": False, "note": f"共同标的仅 {len(common)} 个，至少需要 3 个"}
    a = pd.Series({c: scores[c] for c in common}, dtype=float)
    b = pd.Series({c: forward_returns[c] for c in common}, dtype=float)
    ic = a.corr(b, method="pearson")
    rank_ic = a.corr(b, method="spearman")
    t_obs = len(common)
    se = 1.0 / np.sqrt(max(t_obs - 1, 1))
    return {
        "available": True,
        "n": t_obs,
        "ic": round(float(ic), 4) if pd.notna(ic) else None,
        "rank_ic": round(float(rank_ic), 4) if pd.notna(rank_ic) else None,
        "t_stat": round(float(rank_ic) / se, 2) if pd.notna(rank_ic) else None,
    }


DEFAULT_SCORE_FN: Callable[[pd.DataFrame], Dict[str, float]] = \
    lambda window: {c: round(annualized_return(window[c]) * 100, 3) for c in window.columns
                    if len(window[c].dropna()) > 1}


def walk_forward(nav_by_code: Dict[str, pd.Series],
                 train_days: int = 252, test_days: int = 63, folds: int = 4,
                 score_fn: Optional[Callable[[pd.DataFrame], Dict[str, float]]] = None,
                 top_n: int = 3) -> Dict[str, object]:
    """
    滚动样本外检验打分规则的有效性。

    每一折：用 [t-train_days, t) 的净值算分 -> 看 [t, t+test_days) 的真实收益，
    统计 Rank IC、头部命中率（选出的 top_n 是否真跑赢中位数）。

    nav_by_code 需要覆盖 (train_days + test_days * folds) 以上的历史，否则自动减少折数。
    """
    frame = pd.DataFrame({c: pd.Series(s.values, index=s.index) for c, s in nav_by_code.items()})
    frame = frame.sort_index().dropna(how="all").ffill()
    score_fn = score_fn or DEFAULT_SCORE_FN

    total = len(frame)
    need = train_days + test_days
    if total < need + 20 or frame.shape[1] < 2:
        return {
            "available": False,
            "note": f"历史长度不足：需要至少 {need} 个交易日与 2 只基金，"
                    f"当前 {total} 天 / {frame.shape[1]} 只",
            "observations": total,
        }

    usable_folds = min(int(folds), max((total - train_days) // test_days, 1))
    records: List[dict] = []
    picks: List[str] = []

    for k in range(usable_folds):
        test_end = total - k * test_days
        test_start = test_end - test_days
        train_start = test_start - train_days
        if train_start < 0:
            break
        train_window = frame.iloc[train_start:test_start]
        test_window = frame.iloc[test_start:test_end]
        if len(train_window) < int(train_days * 0.6) or len(test_window) < 2:
            continue

        scores = score_fn(train_window)
        if len(scores) < 3:
            continue

        fwd = {}
        for code in test_window.columns:
            s = test_window[code].dropna()
            if len(s) > 1:
                fwd[code] = float(s.iloc[-1] / s.iloc[0] - 1) * 100
        ic = information_coefficient(scores, fwd)
        if not ic.get("available"):
            continue

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        chosen = [c for c, _ in ranked[:top_n]]
        median_fwd = float(np.median(list(fwd.values()))) if fwd else 0.0
        beats = [c for c in chosen if c in fwd and fwd[c] > median_fwd]
        picks.append([c for c in chosen])
        records.append({
            "fold": usable_folds - k,
            "train_end": str(train_window.index[-1].date()),
            "test_end": str(test_window.index[-1].date()),
            "rank_ic": ic["rank_ic"],
            "hit_rate_pct": round(len(beats) / len(chosen) * 100, 1) if chosen else None,
            "picked": beats,
            "top_forward_return_pct": round(max([fwd[c] for c in chosen if c in fwd], default=None) or 0.0, 2)
            if chosen else None,
        })

    if not records:
        return {"available": False, "note": "没有任何一折能构成有效检验", "observations": total}

    ics = [r["rank_ic"] for r in records if r["rank_ic"] is not None]
    hits = [r["hit_rate_pct"] for r in records if r["hit_rate_pct"] is not None]
    mean_ic = float(np.mean(ics)) if ics else 0.0
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    t_stat = mean_ic / (ic_std / np.sqrt(len(ics))) if ic_std > EPS else None

    positive = mean_ic > 0.02 and (t_stat is None or t_stat > 1.65)
    return {
        "available": True,
        "folds": len(records),
        "train_days": train_days,
        "test_days": test_days,
        "mean_rank_ic": round(mean_ic, 4),
        "ic_std": round(ic_std, 4),
        "ic_t_stat": round(t_stat, 2) if t_stat is not None else None,
        "mean_hit_rate_pct": round(float(np.mean(hits)), 1) if hits else None,
        "effective_out_of_sample": bool(positive),
        "records": records,
        "interpretation": (
            "样本外仍有正向预测力，排名信息可作为参考" if positive else
            f"样本外平均 Rank IC={mean_ic:.3f}，接近随机：这个排名规则对未来没有可依赖的预测力"
        ),
    }
