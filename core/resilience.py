"""外部调用韧性：限流、重试、熔断，以及把阻塞调用挪出事件循环。

免费公开数据接口有两个现实约束：有速率限制、返回结构会变。
所以这三件事必须集中在一处，而不是散落在每个抓取函数里写 try/except。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from core.errors import RateLimited, SourceUnavailable

logger = logging.getLogger("fund-agent")


class RateLimiter:
    """保证相邻两次外部调用之间至少间隔 min_interval 秒。"""

    def __init__(self, min_interval: float = 0.25):
        self.min_interval = float(min_interval)
        self._last = 0.0
        self._lock = threading.Lock()
        self.waited = 0

    def acquire(self) -> float:
        """阻塞到允许下一次调用的时间，返回实际等待秒数。"""
        if self.min_interval <= 0:
            return 0.0
        with self._lock:
            now = time.time()
            delta = now - self._last
            if delta < self.min_interval:
                sleep_for = self.min_interval - delta
                time.sleep(sleep_for)
                self.waited += 1
                self._last = time.time()
                return sleep_for
            self._last = now
            return 0.0

    def stats(self) -> dict:
        return {"min_interval": self.min_interval, "waits": self.waited}


class CircuitBreaker:
    """
    连续失败 threshold 次后打开，冷却期内直接走降级路径（返回缓存）。

    目的：上游挂了时不要让每个请求都去撞一次网络，导致接口集体超时。
    """

    def __init__(self, threshold: int = 3, reset_timeout: float = 120.0):
        self.threshold = int(threshold)
        self.reset_timeout = float(reset_timeout)
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if (time.time() - self._opened_at) >= self.reset_timeout:
                # 冷却结束，半开：允许下一次探测
                self._opened_at = None
                self._failures = 0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold and self._opened_at is None:
                self._opened_at = time.time()
                logger.warning(f"[熔断] 连续失败 {self._failures} 次，打开熔断 {self.reset_timeout}s")

    @property
    def failures(self) -> int:
        return self._failures

    def state(self) -> str:
        return "open" if self.open else "closed"


class BreakerRegistry:
    """
    按接口名分别持有熔断器。

    一个全局熔断器会让"某只基金代码不存在"这类数据级失败，
    连带把其它正常请求一起挡在门外。熔断的粒度必须是接口，不是整个应用。
    """

    def __init__(self, threshold: int = 3, reset_timeout: float = 120.0, max_keys: int = 16):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.max_keys = max_keys
        self._items: "OrderedDict[str, CircuitBreaker]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._items.get(key)
            if breaker is None:
                breaker = CircuitBreaker(self.threshold, self.reset_timeout)
                self._items[key] = breaker
                self._items.move_to_end(key)
                while len(self._items) > self.max_keys:
                    self._items.popitem(last=False)
            return breaker

    def state_for(self, key: str) -> str:
        with self._lock:
            breaker = self._items.get(key)
        return breaker.state() if breaker else "closed"

    def open_keys(self) -> List[str]:
        with self._lock:
            items = list(self._items.items())
        return [k for k, b in items if b.open]

    def state(self) -> str:
        return "open" if self.open_keys() else "closed"

    def summary(self) -> Dict[str, str]:
        with self._lock:
            items = list(self._items.items())
        return {k: b.state() for k, b in items}


# 网络/上游级异常：应当计入熔断并重试
_INFRA_MARKERS = ("Connection", "Timeout", "SSLError", "Proxy", "HTTPError",
                  "RemoteDisconnected", "ChunkedEncoding", "NameResolution", "ReadTimeout")
# 解析/数据级异常（代码不存在、返回结构变化）：重试没有意义，也不该熔断整个接口
_DATA_MARKERS = ("TypeError", "KeyError", "ValueError", "AttributeError",
                 "IndexError", "ZeroDivisionError", "EmptyDataError")


def is_infrastructure_error(exc: BaseException) -> bool:
    """
    区分"上游挂了"和"这条数据没有"。

    akshare 的异常类型很杂，这里按异常类名（含继承链）判断；
    未知类型保守地当作上游故障处理。
    """
    for klass in type(exc).__mro__:
        name = klass.__name__
        if any(marker in name for marker in _DATA_MARKERS):
            return False
        if any(marker in name for marker in _INFRA_MARKERS):
            return True
    module = type(exc).__module__ or ""
    if "requests" in module or "urllib" in module or "http" in module or "socket" in module:
        return True
    return True


def call_with_retry(fn: Callable[..., Any], *args,
                    retries: int = 2, backoff: float = 0.6,
                    limiter: Optional[RateLimiter] = None,
                    breaker: Optional[CircuitBreaker] = None,
                    label: str = "external", **kwargs) -> Any:
    """
    同步外部调用统一入口：限流 -> 重试（指数退避）-> 记录熔断状态。

    只有网络级失败才重试并计入熔断；数据级失败（代码不存在、结构变化）
    立即返回，避免"一只垃圾代码拖垮整个接口"。
    抛出 SourceUnavailable，由上层决定降级方式（通常是读缓存）。
    """
    if breaker and breaker.open:
        raise SourceUnavailable(f"{label} 正在熔断冷却中，已跳过外部调用")

    attempts = max(1, int(retries) + 1)
    last_err: Optional[Exception] = None
    for attempt in range(attempts):
        if limiter:
            limiter.acquire()
        try:
            out = fn(*args, **kwargs)
            if breaker:
                breaker.record_success()
            return out
        except Exception as e:  # akshare 异常类型不固定，统一兜住后分类
            last_err = e
            if not is_infrastructure_error(e):
                if breaker:
                    breaker.record_success()      # 不是上游问题，不影响熔断计数
                logger.warning(f"[{label}] 数据级失败（不重试、不熔断）: {type(e).__name__}: {e}")
                raise SourceUnavailable(f"{label} 无有效数据: {type(e).__name__}: {e}") from e
            if breaker:
                breaker.record_failure()
            if attempt < attempts - 1:
                sleep_for = backoff * (2 ** attempt)
                logger.warning(f"[{label}] 第 {attempt + 1} 次失败: {e}，{sleep_for:.1f}s 后重试")
                time.sleep(sleep_for)

    if breaker and breaker.open:
        raise SourceUnavailable(f"{label} 连续失败已熔断: {last_err}")
    raise SourceUnavailable(f"{label} 调用失败: {last_err}")


async def run_blocking(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """
    在默认线程池里跑阻塞函数。

    akshare 是同步的，直接在 async 路由里调用会卡住整个事件循环。
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def raise_rate_limited(message: str) -> None:
    raise RateLimited(message)
