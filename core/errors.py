"""统一异常类型。

路由层不再逐个 try/except -> HTTPException(500)，而是让业务异常自然抛出，
由 register_exception_handlers 统一翻译成 HTTP 响应。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("fund-agent")


class AppError(Exception):
    """业务异常基类。"""
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str = "", *, extra: Optional[Dict[str, Any]] = None):
        super().__init__(message or self.__class__.__doc__ or "内部错误")
        self.message = message or "内部错误"
        self.extra = extra or {}

    def to_payload(self) -> Dict[str, Any]:
        # detail 字段是为了兼容旧前端读取 FastAPI 的 {detail: ...}
        return {"error": self.code, "message": self.message,
                "detail": self.message, **self.extra}


class InvalidInput(AppError):
    """入参不合法（格式、范围、数量）。"""
    status_code = 400
    code = "invalid_input"


class InsufficientData(AppError):
    """样本量或对齐数据不足以得出结论。"""
    status_code = 422
    code = "insufficient_data"


class SourceUnavailable(AppError):
    """外部数据源不可用且降级缓存也不可用。"""
    status_code = 503
    code = "source_unavailable"


class RateLimited(AppError):
    """触发上游或本地限流。"""
    status_code = 429
    code = "rate_limited"


class LLMUnavailable(AppError):
    """LLM 未配置或调用失败。"""
    status_code = 503
    code = "llm_unavailable"


class NotFound(AppError):
    """资源不存在。"""
    status_code = 404
    code = "not_found"


def register_exception_handlers(app: FastAPI) -> None:
    """把 AppError 映射为结构化 JSON 响应。"""

    @app.exception_handler(AppError)
    async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error(f"[{exc.code}] {exc.message}")
        else:
            logger.warning(f"[{exc.code}] {exc.message}")
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(Exception)
    async def _unexpected_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"[unhandled] {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": f"服务内部错误: {exc}"},
        )


def raise_invalid(*args, **kwargs) -> None:
    raise InvalidInput(*args, **kwargs)
