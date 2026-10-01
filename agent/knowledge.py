"""内置基金知识库检索。

关键词命中打分（够用且完全离线）。换成向量检索时只需要替换这个文件里的
`search` 实现，上层调用点不变 —— 这就是把它单独放一个模块的原因。
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("fund-agent")

DISCLAIMER = "\n\n（以上内容来自内置基金知识库，仅供参考，不构成投资建议。）"


class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: Optional[List[dict]] = None
        self._lock = threading.Lock()

    # ---- 加载 ----
    def load(self, force: bool = False) -> List[dict]:
        with self._lock:
            if self._entries is not None and not force:
                return self._entries
            if not self.path.exists():
                logger.warning(f"[知识库] 文件不存在: {self.path}")
                self._entries = []
                return self._entries
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._entries = json.load(f)
                logger.info(f"[知识库] 已加载 {len(self._entries)} 条知识条目")
            except Exception as e:
                logger.error(f"[知识库] 加载失败: {e}")
                self._entries = []
            return self._entries

    @property
    def count(self) -> int:
        return len(self.load())

    # ---- 检索 ----
    @staticmethod
    def score(query: str, entry: dict) -> int:
        q = query.strip().lower()
        s = 0
        for kw in entry.get("keywords", []):
            kw_l = str(kw).lower()
            if kw_l and (kw_l in q or q in kw_l):
                s += 3
        title = str(entry.get("title", "")).lower()
        if title and (title in q or q in title):
            s += 4
        if str(entry.get("category", "")).lower() in q and q:
            s += 1
        return s

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        entries = self.load()
        q = (query or "").strip().lower()
        if not entries or not q:
            return []
        scored = [(self.score(query, e), e) for e in entries]
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def answer(self, query: str, top_k: int = 2, min_score: int = 3) -> Optional[str]:
        """命中则返回格式化答案，否则 None（让上层继续降级）。"""
        hits = [e for e in self.search(query, top_k=top_k) if self.score(query, e) >= min_score]
        if not hits:
            return None
        parts = [f"【{e.get('title', '')}】\n{e.get('content', '')}" for e in hits]
        return "\n\n———\n\n".join(parts) + DISCLAIMER

    def categories(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for e in self.load():
            out.setdefault(e.get("category", "其他"), []).append(e.get("title"))
        return out


_default: Optional[KnowledgeBase] = None


def get_knowledge_base(path: Optional[Path] = None) -> KnowledgeBase:
    """默认单例（组合根 container 会用显式实例覆盖它）。"""
    global _default
    if _default is None or path is not None:
        from config import KNOWLEDGE_FILE
        _default = KnowledgeBase(path or KNOWLEDGE_FILE)
    return _default


# ---- 兼容旧的模块级函数调用 ----
def load_knowledge_base(force: bool = False) -> List[dict]:
    return get_knowledge_base().load(force=force)


def search_knowledge(query: str, top_k: int = 3) -> List[dict]:
    return get_knowledge_base().search(query, top_k=top_k)


def format_knowledge_answer(query: str) -> Optional[str]:
    return get_knowledge_base().answer(query)
