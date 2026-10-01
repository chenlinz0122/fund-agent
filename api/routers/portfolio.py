"""组合分析路由：指标/相关性、优化、回测、稳健性体检、费后期望。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.deps import state
from api.schemas import BacktestRequest, ExpectancyRequest, PortfolioRequest, RobustnessRequest
from core.resilience import run_blocking
from services import portfolio as pf

router = APIRouter(tags=["portfolio"], prefix="/api/portfolio")


@router.post("/analyze")
async def analyze(request: PortfolioRequest, st=Depends(state)):
    """A 风险指标 + D 多因子打分 + E 相关性诊断（含有效下注数）。"""
    return await run_blocking(pf.analyze, st.repo, st.settings, request.codes, request.days)


@router.post("/optimize")
async def optimize(request: PortfolioRequest,
                   target_vol: Optional[float] = Query(None, ge=0.01, le=1.0,
                                                       description="目标年化波动率，留空用配置默认值"),
                   st=Depends(state)):
    """B 组合优化：最大夏普/最小方差/风险平价 + 有效前沿 + 无杠杆护栏。"""
    return await run_blocking(pf.optimize, st.repo, st.settings,
                              request.codes, request.days, target_vol)


@router.post("/backtest")
async def backtest(request: BacktestRequest, st=Depends(state)):
    """C 组合回测：净值曲线 + 指标 + 对比基准，可计入换手成本。"""
    return await run_blocking(pf.run_backtest, st.repo, st.settings, request.codes,
                              request.days, request.weights, request.rebalance, request.cost_pct)


@router.post("/robustness")
async def robustness(request: RobustnessRequest, st=Depends(state)):
    """
    反过拟合体检：walk-forward 样本外有效性 + Deflated Sharpe。

    回答的是"这套排名到底有没有预测力"，而不是"它过去多好看"。
    """
    return await run_blocking(
        pf.robustness, st.repo, st.settings, request.codes, request.days,
        request.train_days, request.test_days, request.folds, request.n_trials)


@router.post("/expectancy")
async def expectancy(request: ExpectancyRequest, st=Depends(state)):
    """费后期望对照：毛收益 vs 费用侵蚀 vs 净收益 vs 净夏普。"""
    return await run_blocking(
        pf.expectancy_batch, st.repo, st.settings, request.codes, request.days,
        request.round_trips_per_year, request.fund_types)
