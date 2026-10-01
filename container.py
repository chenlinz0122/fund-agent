"""组合根（composition root）：整个应用唯一持有"全局实例"的地方。

原来的 main.py 里散落着 `_cache / _nav_cache / _bench_cache / LLM_STATE / _knowledge_base`
五个模块级可变全局，谁都能改、测试之间互相污染、也看不出依赖关系。
现在它们收敛成唯一一个 AppState，在启动时构造一次，通过依赖注入传给路由。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from agent.intent import IntentRouter
from agent.knowledge import KnowledgeBase
from agent.llm import LLMClient
from agent.service import ChatService
from analysis.archive import StrategyArchive
from config import Settings, load_settings
from paper.store import PaperStore
from sources.akshare_client import AkshareClient
from sources.repository import FundRepository

logger = logging.getLogger("fund-agent")


@dataclass
class AppState:
    settings: Settings
    client: AkshareClient
    repo: FundRepository
    kb: KnowledgeBase
    llm: LLMClient
    router: IntentRouter
    paper: PaperStore = field(default_factory=PaperStore)
    archive: StrategyArchive = field(default_factory=StrategyArchive)
    chat: ChatService = field(default=None)  # 下面 __post_init__ 里构造

    def __post_init__(self):
        self.chat = ChatService(self.settings, self.repo, self.kb, self.llm, self.router)

    def warmup(self) -> dict:
        """启动预热：知识库 + 排名数据。外部数据不可用时不阻塞启动。"""
        report = {"knowledge": self.kb.load(force=True) and self.kb.count, "ranking": None, "warning": None}
        try:
            df = self.repo.ranking()
            report["ranking"] = int(len(df))
        except Exception as e:
            report["warning"] = f"排名数据预热失败，将在首次请求时重试: {e}"
            logger.warning(report["warning"])
        return report

    def health(self) -> dict:
        return {
            "status": "ok",
            "llm_configured": self.llm.enabled,
            "llm_model": self.llm.model,
            "knowledge_count": self.kb.count,
            "cache": self.repo.health(),
            "config": self.settings.to_dict(),
        }


_state: Optional[AppState] = None
_lock = threading.Lock()


def build_state(settings: Optional[Settings] = None) -> AppState:
    settings = settings or load_settings()
    client = AkshareClient(settings.source)
    repo = FundRepository(settings, client)
    kb = KnowledgeBase(settings.knowledge_file)
    llm = LLMClient(settings.llm)
    logger.info(
        f"[启动] LLM {'已配置: ' + llm.model if llm.enabled else '未配置（使用规则引擎）'}")
    return AppState(settings=settings, client=client, repo=repo, kb=kb,
                    llm=llm, router=IntentRouter())


def get_state() -> AppState:
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = build_state()
    return _state


def set_state(state: AppState) -> None:
    global _state
    _state = state


def reset_state() -> None:
    global _state
    _state = None
