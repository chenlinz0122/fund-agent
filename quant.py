"""
quant.py —— 兼容用的薄转发层（已废弃）。

原来这里是 499 行的"什么都装"的模块。现在按职责拆进 analysis/：

    analysis/metrics.py       风险/收益指标（年化、回撤、Sharpe/Sortino/Calmar、VaR/CVaR、Beta/Alpha）
    analysis/correlation.py   相关性矩阵、高相关预警、有效下注数
    analysis/portfolio.py     组合优化（最小方差/最大夏普/风险平价/有效前沿）+ 无杠杆与不做空护栏
    analysis/backtest.py      组合回测（含真实月度再平衡与换手成本）
    analysis.factors.py       多因子 Z-score 综合打分
    analysis/expectancy.py    费后净期望、收益集中度、结构性负期望检查
    analysis/validation.py    walk-forward 样本外检验、Deflated Sharpe

新代码请 `from analysis import ...`。本文件只保证旧调用点与旧测试继续可用。
"""
from __future__ import annotations

import warnings

from analysis.backtest import backtest
from analysis.constants import DEFAULT_RISK_FREE, EPS, TRADING_DAYS
from analysis.correlation import (
    build_returns_matrix, correlation_matrix, effective_number_of_bets, high_correlation_pairs,
)
from analysis.expectancy import (
    expectancy_report, fee_drag, fee_profile, luck_concentration, negative_expectancy_checks,
)
from analysis.factors import multi_factor_score
from analysis.metrics import (
    annualized_return, annualized_volatility, beta_alpha, calmar_ratio, cvar_95,
    max_drawdown, risk_metrics, sharpe_ratio, sortino_ratio, to_returns, var_95, win_rate,
)
from analysis.portfolio import (
    efficient_frontier, leverage_guard, max_sharpe_weights, min_variance_weights,
    optimize_portfolio, risk_parity_weights, vol_target_weights, weights_dict,
)
from analysis.validation import (
    deflated_sharpe_ratio, expected_max_sharpe, information_coefficient,
    sharpe_summary, walk_forward,
)

# 兼容旧别名
_EPS = EPS

warnings.warn(
    "quant 模块已拆分为 analysis/*，请改用 `from analysis import ...`；"
    "quant.py 仅作为兼容转发层保留。",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "annualized_return", "annualized_volatility", "backtest", "beta_alpha", "build_returns_matrix",
    "calmar_ratio", "correlation_matrix", "cvar_95", "deflated_sharpe_ratio",
    "effective_number_of_bets", "efficient_frontier", "expectancy_report", "expected_max_sharpe",
    "fee_drag", "fee_profile", "high_correlation_pairs", "information_coefficient",
    "leverage_guard", "luck_concentration", "max_drawdown", "max_sharpe_weights",
    "min_variance_weights", "multi_factor_score", "negative_expectancy_checks", "optimize_portfolio",
    "risk_metrics", "risk_parity_weights", "sharpe_ratio", "sharpe_summary", "sortino_ratio",
    "to_returns", "var_95", "vol_target_weights", "walk_forward", "weights_dict", "win_rate",
]
