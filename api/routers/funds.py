"""基金数据路由：排名、筛选、详情、导出。"""
from __future__ import annotations

import io
import json
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from api.deps import state
from api.schemas import ScreenRequest
from core.resilience import run_blocking
from services import funds as funds_service
from services import screening as scr

router = APIRouter(tags=["funds"])


@router.get("/api/funds")
async def get_funds(refresh: bool = Query(False, description="强制刷新缓存"), st=Depends(state)):
    """基金排名数据（前 200 条）。"""
    df = await run_blocking(st.repo.ranking, force_refresh=refresh)
    head = df.head(200)
    # JSON 不接受 NaN（旧实现也是在这里做一次替换），必须在出参前处理
    head = head.astype(object).where(pd.notna(head), None)
    return {"funds": head.to_dict(orient="records"), "total": len(head),
            "cache": st.repo.ranking_meta()}


@router.post("/api/screen")
async def screen_funds(request: ScreenRequest, st=Depends(state)):
    """按条件筛选基金，并如实报告哪些条件真正生效。"""
    df = await run_blocking(st.repo.ranking)
    result = await run_blocking(scr.apply_criteria, df, request.to_criteria())
    return result.to_dict()


@router.get("/api/fund/{code}")
async def fund_detail(code: str, days: int = Query(365, ge=30, le=2000),
                      refresh: bool = Query(False), st=Depends(state)):
    """基金详情：净值历史 + 风险指标 + 费后期望。"""
    return await run_blocking(funds_service.fund_detail, code, st.repo, st.settings,
                              days, refresh)


@router.get("/api/export")
async def export_funds(format: str = Query("csv", pattern="^(csv|json)$"),
                       limit: int = Query(500, ge=1, le=2000),
                       fund_type: Optional[str] = None,
                       min_return_1y: Optional[float] = None,
                       max_return_1y: Optional[float] = None,
                       sort_by: str = "return_1y",
                       sort_order: str = "desc",
                       st=Depends(state)):
    """导出筛选后的基金数据（CSV/JSON）。"""
    df = await run_blocking(st.repo.ranking)
    criteria = scr.FundCriteria(fund_type=fund_type, min_return_1y=min_return_1y,
                                max_return_1y=max_return_1y, sort_by=sort_by,
                                sort_order=sort_order, limit=limit)
    result = await run_blocking(scr.apply_criteria, df, criteria)

    if format == "json":
        content = json.dumps(result.funds, ensure_ascii=False, indent=2, default=str)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=funds_export.json"},
        )

    csv_bytes = scr.to_csv(result.funds).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=funds_export.csv"},
    )
