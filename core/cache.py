"""缓存原语。

替换原先散落在 main.py 里的 `_cache / _nav_cache / _bench_cache` 全局字典：
所有缓存都必须是显式实例、带 TTL、带容量上限，并且可以被注入和替换。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

logger = logging.getLogger("fund-agent")

T = TypeVar("T")


class TTLCache(Generic[T]):
    """线程安全的 LRU + TTL 缓存。"""

    def __init__(self, ttl: float, maxsize: int = 256):
        self.ttl = float(ttl)
        self.maxsize = int(maxsize)
        self._data: "OrderedDict[Any, Tuple[float, T]]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[T]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            ts, value = entry
            if (time.time() - ts) >= self.ttl:   # ttl<=0 表示不缓存
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: Any, value: T, timestamp: Optional[float] = None) -> None:
        with self._lock:
            self._data[key] = (timestamp if timestamp is not None else time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def pop(self, key: Any) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def age(self, key: Any) -> Optional[float]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            return round(time.time() - entry[0], 1)

    def get_or_load(self, key: Any, loader: Callable[[], T]) -> T:
        """读缓存，未命中则调用 loader 并回填。"""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = loader()
        if value is not None:
            self.set(key, value)
        return value

    @property
    def size(self) -> int:
        return len(self._data)

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": self.size,
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }


class DiskCache:
    """基于文件 mtime 的磁盘缓存（JSON / pandas）。"""

    def __init__(self, path: Path, ttl: float):
        self.path = Path(path)
        self.ttl = float(ttl)

    def exists(self) -> bool:
        return self.path.exists()

    def fresh(self) -> bool:
        return self.path.exists() and (time.time() - self.path.stat().st_mtime) < self.ttl

    def age(self) -> Optional[float]:
        if not self.path.exists():
            return None
        return round(time.time() - self.path.stat().st_mtime, 1)

    def read_json(self) -> Optional[Any]:
        if not self.path.exists():
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # 缓存损坏不应导致接口 500
            logger.warning(f"[磁盘缓存] 读取失败 {self.path.name}: {e}")
            return None

    def read_dataframe(self):
        import pandas as pd
        if not self.path.exists():
            return None
        try:
            return pd.read_json(self.path, orient="records")
        except Exception as e:
            logger.warning(f"[磁盘缓存] 读取 DataFrame 失败 {self.path.name}: {e}")
            return None

    def write_json(self, payload: Any) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.warning(f"[磁盘缓存] 写入失败 {self.path.name}: {e}")
            return False

    def write_dataframe(self, df) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            df.to_json(self.path, orient="records", force_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.warning(f"[磁盘缓存] 写入 DataFrame 失败 {self.path.name}: {e}")
            return False

    def invalidate(self) -> None:
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception as e:
            logger.warning(f"[磁盘缓存] 删除失败 {self.path.name}: {e}")


class MemoryCache:
    """
    兼容旧接口的门面（原 main.MemoryCache）。

    新代码请用 TTLCache / DiskCache；这里保留 get_ranking / get_detail 等方法，
    以免已有调用点和测试立刻失效。
    """

    def __init__(self, ttl: float = 3600 * 4, detail_ttl: float = 3600):
        self.ttl = ttl
        self.detail_ttl = detail_ttl
        self._ranking: Optional[Any] = None
        self._ranking_ts: float = 0.0
        self._details = TTLCache(ttl=detail_ttl, maxsize=512)

    def get_ranking(self):
        if self._ranking is not None and (time.time() - self._ranking_ts) < self.ttl:
            return self._ranking
        return None

    def set_ranking(self, df, timestamp: Optional[float] = None) -> None:
        self._ranking = df
        self._ranking_ts = timestamp if timestamp is not None else time.time()

    @property
    def ranking_age(self) -> Optional[float]:
        if self._ranking is None or self._ranking_ts == 0:
            return None
        return round(time.time() - self._ranking_ts, 1)

    def get_detail(self, code: str):
        return self._details.get(code)

    def set_detail(self, code: str, data: dict) -> None:
        self._details.set(code, data)

    @property
    def detail_count(self) -> int:
        return self._details.size

    def invalidate_ranking(self) -> None:
        self._ranking = None
        self._ranking_ts = 0.0

    def invalidate_detail(self, code: str) -> None:
        self._details.pop(code)

    def invalidate_all(self) -> None:
        self.invalidate_ranking()
        self._details.clear()
