"""问答编排服务：意图路由 -> 规则答案 -> （可选）LLM 答案 -> 降级。

优先级设计：能用规则确定回答的（榜单、筛选、对比、费后计算）不消耗 token；
只有概念解释、开放式追问才交给 LLM；LLM 失败或预算耗尽自动回到规则引擎。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from agent import prompts, rules
from agent.intent import IntentRouter
from agent.knowledge import KnowledgeBase
from agent.llm import LLMClient
from config import Settings
from core.errors import LLMUnavailable
from sources import normalize
from sources.repository import FundRepository

logger = logging.getLogger("fund-agent")


class ChatService:
    def __init__(self, settings: Settings, repo: FundRepository,
                 kb: KnowledgeBase, llm: LLMClient, router: Optional[IntentRouter] = None):
        self.settings = settings
        self.repo = repo
        self.kb = kb
        self.llm = llm
        self.router = router or IntentRouter()
        self.ctx = rules.AgentContext(settings=settings, repo=repo, kb=kb)

    # ---------- 规则路径 ----------
    def rule_answer(self, message: str, df: Optional[pd.DataFrame] = None) -> str:
        for name in self.router.match(message):
            handler = rules.HANDLERS.get(name)
            if not handler:
                continue
            try:
                answer = handler(message, df, self.ctx)
            except Exception as e:
                logger.warning(f"[规则] {name} 处理异常: {e}")
                continue
            if answer:
                return answer
        kb_answer = self.kb.answer(message)
        return kb_answer or rules.default_help()

    # ---------- 数据概况（给 LLM 的上下文） ----------
    def data_stats(self, df: Optional[pd.DataFrame]) -> dict:
        stats = {"as_of": datetime.now().strftime("%Y-%m-%d %H:%M")}
        if df is None or df.empty:
            return stats
        col_map = normalize.resolve_columns(df)
        stats["fund_count"] = int(len(df))
        col = col_map.get("return_1y")
        if col:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series):
                stats["avg_return_1y"] = round(float(series.mean()), 2)
                stats["max_return_1y"] = round(float(series.max()), 2)
        return stats

    def build_messages(self, message: str, history: list, df: Optional[pd.DataFrame]) -> list:
        stats = self.data_stats(df)
        context = prompts.build_context(stats)
        hits = self.kb.search(message, top_k=3)
        if hits:
            extra = "\n【相关基金知识参考】\n" + "\n".join(
                f"- {h.get('title')}：{(h.get('content') or '')[:300]}" for h in hits)
            context += extra
        context = context[: self.settings.llm.context_chars]

        messages = [{"role": "system", "content": prompts.system_prompt(context)}]
        for h in (history or [])[-self.settings.llm.history_limit:]:
            if isinstance(h, dict) and h.get("role") and h.get("content"):
                messages.append({"role": h["role"], "content": str(h["content"])[:2000]})
        messages.append({"role": "user", "content": message})
        return messages

    # ---------- 对外主入口 ----------
    async def answer(self, message: str, history: Optional[list] = None,
                     df: Optional[pd.DataFrame] = None,
                     prefer_llm: bool = True) -> dict:
        """返回 {reply, engine, intent, degraded_from}。"""
        if df is None:
            try:
                df = self.repo.ranking()
            except Exception as e:
                logger.warning(f"[问答] 排名数据不可用: {e}")
                df = None

        intent = self.router.route(message)

        if prefer_llm and self.llm.enabled and not self.llm.budget_exhausted():
            try:
                content = await self.llm.chat(self.build_messages(message, history or [], df))
                return {"reply": content + prompts.RESPONSE_GUARDRAIL,
                        "engine": "llm", "intent": intent, "degraded_from": None}
            except LLMUnavailable as e:
                logger.info(f"[问答] LLM 不可用（{e}），改用规则引擎")
                return {"reply": self.rule_answer(message, df), "engine": "rules",
                        "intent": intent, "degraded_from": "llm"}

        return {"reply": self.rule_answer(message, df), "engine": "rules",
                "intent": intent, "degraded_from": None}


def rule_based_agent(message: str, df: Optional[pd.DataFrame] = None) -> str:
    """兼容旧签名（main.analyze -> rule_based_agent）。使用组合根里的默认状态。"""
    from container import get_state
    return get_state().chat.rule_answer(message, df)
