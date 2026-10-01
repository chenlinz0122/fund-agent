"""analysis：纯计算层。

无 I/O、无全局可变状态、不 import akshare / fastapi / config 之外的东西。
这正是"用 Vibe Coding 维护复杂系统"的关键：这一层的任何函数都可以
直接喂给模型做上下文，并用一组固定的 numpy 数组单测验证。
"""
from analysis.constants import DEFAULT_RISK_FREE, TRADING_DAYS  # noqa: F401
from analysis.metrics import (  # noqa: F401
    annualized_return, annualized_volatility, beta_alpha, calmar_ratio,
    max_drawdown, risk_metrics, sharpe_ratio, sortino_ratio, to_returns,
    var_95, win_rate,
)
from analysis.correlation import (  # noqa: F401
    build_returns_matrix, correlation_matrix, effective_number_of_bets,
    high_correlation_pairs,
)
from analysis.portfolio import (  # noqa: F401
    efficient_frontier, leverage_guard, max_sharpe_weights, min_variance_weights,
    optimize_portfolio, risk_parity_weights, vol_target_weights, weights_dict,
)
from analysis.backtest import backtest  # noqa: F401
# 注意：上面的 `backtest` 会把同名子模块 analysis.backtest 的包属性遮蔽成函数。
# 需要模块本身时请用 `from analysis.backtest import backtest as ...`（services 层即如此）。
from analysis.factors import multi_factor_score  # noqa: F401
from analysis.expectancy import (  # noqa: F401
    expectancy_report, fee_drag, fee_profile, negative_expectancy_checks,
)
from analysis.validation import (  # noqa: F401
    deflated_sharpe_ratio, information_coefficient, walk_forward,
)
