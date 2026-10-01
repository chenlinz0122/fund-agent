"""FastAPI 依赖：把组合根 AppState 注入路由。"""
from __future__ import annotations

from fastapi import Request

from container import AppState, get_state


def state(request: Request) -> AppState:
    """优先使用 lifespan 里构造的实例；未走 lifespan 时（例如单测）退回组合根懒加载。"""
    st = getattr(request.app.state, "app_state", None)
    return st if st is not None else get_state()
