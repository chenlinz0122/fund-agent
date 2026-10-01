"""LLM 接入（OpenAI 兼容接口）。

要点：
  - 未配置 Key 时整个类不生效，上层自动退回规则引擎（保持"零配置可用"）；
  - 用量与预算状态集中在这里，是唯一允许在运行期可变的状态；
  - 预算耗尽或调用失败 -> 抛 LLMUnavailable，由 ChatService 决定降级；
  - 只依赖 httpx，不依赖具体厂商，换 base_url 即可用通义/Ollama/其他中转。
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from config import LLMConfig
from core.errors import LLMUnavailable

logger = logging.getLogger("fund-agent")

NON_CHAT_HINTS = ("embedding", "whisper", "moderation", "tts", "dall", "rerank")


def is_chat_model(model_id: str) -> bool:
    m = (model_id or "").lower()
    return not any(h in m for h in NON_CHAT_HINTS)


class LLMClient:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        self.model = self.cfg.model
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.budget_tokens = self.cfg.budget_tokens
        self.last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.api_key)

    # ---- 用量与预算 ----
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def budget_exhausted(self) -> bool:
        with self._lock:
            return bool(self.budget_tokens) and self.total_tokens >= int(self.budget_tokens)

    def set_budget(self, budget: Optional[int]) -> None:
        if budget is not None and budget < 0:
            raise ValueError("预算不能为负")
        with self._lock:
            self.budget_tokens = budget

    def usage(self) -> dict:
        with self._lock:
            total = self.prompt_tokens + self.completion_tokens
            remaining = (int(self.budget_tokens) - total) if self.budget_tokens else None
            return {
                "llm_enabled": self.enabled,
                "model": self.model,
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": total,
                "budget_tokens": self.budget_tokens,
                "remaining_tokens": remaining,
                "balance_api_available": False,
                "last_error": self.last_error,
                "note": "该中转未提供余额查询 API，此处为本地累计 token 用量；剩余额度基于你设定的预算估算。",
            }

    def _record(self, usage: dict, error: Optional[str] = None) -> None:
        with self._lock:
            if error is None:
                self.calls += 1
                self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                self.last_error = None
            else:
                self.last_error = error

    # ---- 模型 ----
    async def list_models(self) -> List[str]:
        import httpx
        if not self.enabled:
            return []
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    f"{self.cfg.api_base}/models",
                    headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                )
                resp.raise_for_status()
                ids = [m.get("id") for m in resp.json().get("data", []) if m.get("id")]
                return [i for i in ids if is_chat_model(i)]
        except Exception as e:
            logger.error(f"[LLM] 获取模型列表失败: {e}")
            self._record({}, error=str(e))
            return []

    def set_model(self, name: str) -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("模型名不能为空")
        with self._lock:
            self.model = name
        return name

    # ---- 调用 ----
    async def chat(self, messages: List[dict], temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None) -> str:
        """一次对话补全；失败/预算耗尽抛 LLMUnavailable，由上层降级到规则引擎。"""
        if not self.enabled:
            raise LLMUnavailable("未配置 LLM_API_KEY")
        if self.budget_exhausted():
            raise LLMUnavailable("token 预算已耗尽，已切换到规则引擎以控制成本")

        import httpx
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
                resp = await client.post(
                    f"{self.cfg.api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.cfg.api_key}",
                             "Content-Type": "application/json"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            self._record(data.get("usage", {}) or {})
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content:
                # 推理型模型可能把正文放在 reasoning_content
                content = (msg.get("reasoning_content") or "").strip()
            if not content:
                raise LLMUnavailable("模型返回空内容")
            return content
        except LLMUnavailable:
            raise
        except Exception as e:
            logger.error(f"[LLM] 调用失败: {e}")
            self._record({}, error=str(e))
            raise LLMUnavailable(f"LLM 调用失败: {e}")
