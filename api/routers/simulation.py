"""策略模拟 / 参数进化 / 风险档位路由。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.deps import state
from api.schemas import EvolveRequest, PolicyRequest, SimulateRequest
from core.resilience import run_blocking
from services import simulation as sim

router = APIRouter(tags=["simulation"], prefix="/api/portfolio")


@router.post("/simulate")
async def simulate_strategy(request: SimulateRequest, st=Depends(state)):
    """按给定基因跑一次策略模拟（状态→仓位→出场→成本）。"""
    return await run_blocking(
        sim.run_simulation, st.repo, st.settings, request.codes, request.days,
        request.genes, request.exit, request.weights, request.season)


@router.post("/evolve")
async def evolve_strategy(request: EvolveRequest, st=Depends(state)):
    """
    遗传搜索。刻意设了评估次数上限：这是网页上点一下要出结果的工具，
    不是云端千万基因实验；分数高低不重要，DSR 与参数平原才决定能不能用。
    """
    return await run_blocking(
        sim.run_evolution, st.repo, st.settings, request.codes, request.days,
        request.pop_size, request.generations, request.seed,
        request.incumbent, request.exit, request.plateau)


@router.get("/policy")
async def policy_snapshot(codes: str = Query(..., description="逗号分隔的基金代码"),
                          days: int = Query(365, ge=90, le=2000), st=Depends(state)):
    """每只基金当前的风险状态与建议敞口档位。"""
    parsed = [c.strip() for c in codes.replace(" ", ",").split(",") if c.strip()]
    return await run_blocking(sim.policy_snapshot, parsed, st.repo, days, None)
