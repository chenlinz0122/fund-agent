"""
基金筛选 Agent —— 应用组装入口。

这个文件只做四件事：配置日志、装配 FastAPI、注册路由与异常处理、启动时构造组合根。
业务逻辑都在分层包里：

    config.py      唯一读取环境变量的地方
    core/          缓存 / 限流 / 熔断 / 异常（无业务）
    sources/       外部数据唯一出入口（只有这里 import akshare）
    analysis/      纯计算：指标、相关性、优化、回测、因子、期望值、样本外检验
    services/      用例编排（sources + analysis 组合出 fund_detail / portfolio 等）
    agent/         问答：意图路由、知识库、LLM、规则处理函数
    api/           HTTP 形状：schemas / deps / routers
    container.py   组合根：唯一的运行期实例集合

改代码前先读 ARCHITECTURE.md 的「要改 X 看哪个文件」表。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import chat, funds, paper, portfolio, simulation, system, training
from config import STATIC_DIR, load_settings
from container import build_state, set_state
from core.errors import register_exception_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fund-agent")

SETTINGS = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时构造 AppState（唯一运行期实例集合）并预热缓存；外部数据不可用不阻塞启动。"""
    state = build_state(SETTINGS)
    app.state.app_state = state
    set_state(state)
    report = state.warmup()
    ranking = f"已预热 {report['ranking']} 条" if report["ranking"] else "待首次请求加载"
    logger.info(f"[启动] 知识库 {report['knowledge']} 条；排名数据 {ranking}")
    if report.get("warning"):
        logger.warning(f"[启动] {report['warning']}")
    logger.info("[启动] 访问地址：http://localhost:8000")
    yield
    logger.info("[关闭] 应用退出")


app = FastAPI(
    title="基金筛选 Agent",
    description="A 股公募基金筛选、组合分析与费后期望体检（akshare 免费数据）",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)

for _router in (system.router, funds.router, portfolio.router, simulation.router,
                chat.router, paper.router, training.router):
    app.include_router(_router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ========== 向后兼容导出 ==========
# 旧代码习惯从 main 引用这些名字（tests/test_main.py 即是一例）。
# 新代码请按层直接导入；本块只是迁移缓冲，确认无外部依赖后可删除。
from agent.intent import (  # noqa: E402  (compat)
    extract_codes,
    extract_fund_type_from_message as _extract_fund_type_from_message,
    extract_return_threshold as _extract_return_threshold,
)
from agent.knowledge import (  # noqa: E402  (compat)
    format_knowledge_answer, load_knowledge_base, search_knowledge,
)
from agent.service import ChatService, rule_based_agent  # noqa: E402  (compat)
from api.schemas import ChatRequest, ScreenRequest  # noqa: E402  (compat)
from container import AppState, get_state  # noqa: E402  (compat)
from core.cache import MemoryCache  # noqa: E402  (compat)
from services.funds import fund_detail as _fund_detail  # noqa: E402  (compat)
from services.screening import analyze_funds, csv_escape as _csv_escape  # noqa: E402  (compat)
from sources.normalize import (  # noqa: E402  (compat)
    COLUMN_MAPPING, build_column_map, resolve_columns, safe_float as _safe_float,
)


def fetch_fund_detail(fund_code: str) -> dict:
    """兼容旧签名：内部使用组合根里的仓库实例。"""
    st = get_state()
    return _fund_detail(fund_code, st.repo, st.settings)


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 50)
    logger.info("基金筛选 Agent 启动中...")
    logger.info(f"LLM 配置：{'已配置' if SETTINGS.llm.enabled else '未配置（使用规则引擎）'}")
    logger.info("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
