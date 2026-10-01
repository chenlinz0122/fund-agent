"""系统路由：首页、健康检查、缓存管理、LLM 模型与预算。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, HTMLResponse

from api.deps import state
from api.schemas import SetBudgetRequest, SetModelRequest
from config import STATIC_DIR
from core.errors import InvalidInput

router = APIRouter(tags=["system"])


@router.get("/")
async def index():
    """单页前端。"""
    html = STATIC_DIR / "index.html"
    if not html.exists():
        return HTMLResponse("<h1>static/index.html 缺失</h1>", status_code=500)
    return FileResponse(html, media_type="text/html")


@router.get("/api/health")
async def health(st=Depends(state)):
    """健康检查。保留旧字段名，额外补充缓存与数据源状态。"""
    repo = st.repo
    return {
        "status": "ok",
        "llm_configured": st.llm.enabled,
        "llm_model": st.llm.model,
        "cache_exists": repo.ranking_disk.exists(),
        "memory_cache_hit": repo.ranking_cached(),
        "cache_age_seconds": repo.ranking_meta()["disk_age_seconds"],
        "detail_cache_size": repo.detail_cache_size(),
        "knowledge_count": st.kb.count,
        "breaker": st.client.breakers.state(),
        "open_breakers": st.client.breakers.open_keys(),
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/api/cache/invalidate")
async def invalidate_cache(st=Depends(state)):
    """清除内存与文件缓存。"""
    st.repo.invalidate_all()
    return {"status": "ok", "message": "缓存已清除"}


# ========== LLM 模型 / 用量 ==========
@router.get("/api/llm/models")
async def llm_models(st=Depends(state)):
    """可用对话模型列表 + 当前模型。"""
    available = await st.llm.list_models()
    current = st.llm.model
    if not st.llm.enabled:
        return {"llm_enabled": False, "available": [], "current": current}
    if current and current not in available:
        available = [current] + available
    return {"llm_enabled": True, "available": available, "current": current,
            "count": len(available)}


@router.post("/api/llm/model")
async def set_model(request: SetModelRequest, st=Depends(state)):
    """运行时切换模型（不改 .env，重启后回到默认）。"""
    name = (request.model or "").strip()
    if not name:
        raise InvalidInput("模型名不能为空")
    allowed = await st.llm.list_models()
    if allowed and name not in allowed:
        raise InvalidInput(f"模型 {name} 不在可用列表中")
    return {"current": st.llm.set_model(name), "ok": True}


@router.get("/api/llm/usage")
async def llm_usage(st=Depends(state)):
    """token 用量与预算剩余。"""
    return st.llm.usage()


@router.post("/api/llm/budget")
async def set_budget(request: SetBudgetRequest, st=Depends(state)):
    """设置 token 预算；传 null 取消。预算耗尽后自动降级到规则引擎。"""
    try:
        st.llm.set_budget(request.budget_tokens)
    except ValueError as e:
        raise InvalidInput(str(e))
    return {"budget_tokens": request.budget_tokens, "ok": True}


@router.get("/api/config")
async def config_view(st=Depends(state)):
    """当前生效配置（只读，用于排查"为什么结果和预期不一样"）。"""
    return st.settings.to_dict()
