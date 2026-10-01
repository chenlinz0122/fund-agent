"""问答与资讯路由。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.deps import state
from api.schemas import ChatRequest
from core.resilience import run_blocking

router = APIRouter(tags=["chat"])


@router.post("/api/chat")
async def chat(request: ChatRequest, st=Depends(state)):
    """智能问答：命中规则能确定的问题不消耗 token，开放式问题才走 LLM。"""
    df = await run_blocking(st.repo.ranking)
    result = await st.chat.answer(request.message, request.history, df,
                                  prefer_llm=request.use_llm)
    return result


@router.get("/api/intent")
async def intent_probe(message: str = Query(..., description="待识别的用户问题"),
                       st=Depends(state)):
    """调试用：看看这句话被路由到了哪个意图。"""
    return st.router.describe(message)


@router.get("/api/news")
async def news(limit: int = Query(12, ge=1, le=30), st=Depends(state)):
    """市场热点与财经新闻。"""
    payload = await run_blocking(st.repo.news_payload)
    items = (payload.get("news") or [])[:limit]
    return {"news": items, "boards": payload.get("boards") or [], "count": len(items)}


@router.get("/api/knowledge")
async def knowledge(query: str = Query("", description="检索关键词"),
                    top_k: int = Query(5, ge=1, le=20), st=Depends(state)):
    """检索基金知识库。"""
    entries = st.kb.load()
    if not query:
        return {"total": len(entries), "categories": st.kb.categories()}
    results = st.kb.search(query, top_k=top_k)
    return {"query": query, "count": len(results), "results": results}
