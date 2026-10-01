"""自训练路由：跑一轮进化、查看档案、人工晋升、复盘报告。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from agent import training
from api.deps import state
from api.schemas import PromoteRequest, TrainRoundRequest
from core.errors import InvalidInput
from core.resilience import run_blocking

router = APIRouter(tags=["training"], prefix="/api/train")


@router.post("/round")
async def run_round(request: TrainRoundRequest, st=Depends(state)):
    """
    跑一轮自我训练：进化 → 三重闸门 → 入库 → 留出段首考。

    留出段在进化时被整段藏起来，所以那是该策略第一次接触"没见过的数据"。
    """
    return await run_blocking(
        lambda: training.run_round(st.repo, st.settings, request.codes,
                                   days=request.days, pop_size=request.pop_size,
                                   generations=request.generations, seed=request.seed,
                                   holdout_days=request.holdout_days, archive=st.archive))


@router.get("/archive")
async def archive_view(st=Depends(state)):
    """冠军 / 挑战者 / 被拒 / 退役 全档案。"""
    records = st.archive.records()
    return {"summary": st.archive.summary(),
            "records": [r.to_dict() for r in records],
            "rule": "进化结果默认只是挑战者；闸门不过直接 rejected，晋升必须人工确认并记录冻结日"}


@router.post("/promote")
async def promote(request: PromoteRequest, st=Depends(state)):
    """人工晋升：设为冠军并记录冻结日。未过三闸门的会被拒绝。"""
    result = st.archive.promote(request.record_id, note=request.note)
    if not result.get("ok"):
        raise InvalidInput(result.get("reason", "晋升失败"),
                           extra={"failures": result.get("failures", [])})
    return result


@router.get("/review")
async def review(account_id: str = Query(..., description="模拟账户 id"),
                 narrate: bool = Query(True, description="是否让 LLM 写一段叙述"),
                 st=Depends(state)):
    """
    复盘报告。数字全部由代码算，LLM 只负责把事实写成话；
    没配 Key 或调用失败时退回模板，数字部分不受影响。
    """
    account = st.paper.get(account_id)
    if account is None:
        raise InvalidInput(f"找不到账户 {account_id}")
    report = await run_blocking(training.build_review, account, st.archive, st.paper)
    if narrate:
        report["narration"] = await training.narrate_review(report, st.settings, st.llm)
    else:
        report["narration"] = {"text": report["facts"].get("结论", ""), "engine": "none"}
    return report
