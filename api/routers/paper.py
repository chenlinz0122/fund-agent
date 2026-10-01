"""模拟盘路由：开户、下单、结算、自动跟踪、决策预览。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.deps import state
from api.schemas import OpenAccountRequest, OrderRequest
from core.resilience import run_blocking
from services import paper as svc

router = APIRouter(tags=["paper"], prefix="/api/paper")


@router.post("/accounts")
async def open_account(request: OpenAccountRequest, st=Depends(state)):
    """开模拟账户。绑定策略时会记录冻结日 —— 此后的净值才算样本外证据。"""
    return await run_blocking(
        lambda: svc.open_account(st.paper, st.repo, st.settings, request.name,
                                 request.capital, request.codes, request.strategy_id,
                                 request.seed_buy, archive=st.archive))


@router.get("/accounts")
async def list_accounts(st=Depends(state)):
    return await run_blocking(svc.list_accounts, st.paper)


@router.get("/accounts/{account_id}")
async def account_detail(account_id: str, days: int = Query(1200, ge=90, le=2500),
                         st=Depends(state)):
    return await run_blocking(svc.account_detail, st.paper, st.repo, account_id, days)


@router.delete("/accounts/{account_id}")
async def delete_account(account_id: str, st=Depends(state)):
    return await run_blocking(svc.delete_account, st.paper, account_id)


@router.post("/accounts/{account_id}/orders")
async def place_order(account_id: str, request: OrderRequest, st=Depends(state)):
    """提交申购/赎回订单（T+1 按未知净值确认）。"""
    return await run_blocking(svc.place_order, st.paper, st.repo, account_id,
                              request.code, request.side, request.amount, request.shares,
                              request.reason)


@router.post("/accounts/{account_id}/settle")
async def settle(account_id: str, on_date: str = Query("", max_length=10), st=Depends(state)):
    """用最新公布的净值结算待确认订单并盯市。"""
    return await run_blocking(svc.settle, st.paper, st.repo, account_id, on_date or None)


@router.post("/accounts/{account_id}/auto")
async def auto_run(account_id: str, st=Depends(state)):
    """让账户按策略规则自己走一天：结算 → 判状态 → 下单。"""
    return await run_blocking(
        lambda: svc.auto_run(st.paper, st.repo, st.settings, account_id, archive=st.archive))


@router.get("/accounts/{account_id}/decision")
async def decision(account_id: str, st=Depends(state)):
    """只算不动：今天规则会怎么决定仓位。"""
    return await run_blocking(
        lambda: svc.preview_decision(st.paper, st.repo, account_id, archive=st.archive))
