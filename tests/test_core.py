"""core 层测试：缓存、限流、熔断、重试、异常映射。全部离线。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from core.cache import DiskCache, MemoryCache, TTLCache
from core.errors import AppError, InsufficientData, InvalidInput, SourceUnavailable
from core.resilience import (
    BreakerRegistry, CircuitBreaker, RateLimiter, call_with_retry,
    is_infrastructure_error, run_blocking,
)


class TestTTLCache:
    def test_hit_and_miss(self):
        c = TTLCache(ttl=60, maxsize=8)
        assert c.get("k") is None
        c.set("k", 1)
        assert c.get("k") == 1
        st = c.stats()
        assert st["hits"] == 1 and st["misses"] == 1

    def test_expiry(self):
        c = TTLCache(ttl=0.01)
        c.set("k", 1)
        time.sleep(0.02)
        assert c.get("k") is None

    def test_ttl_zero_never_caches(self):
        c = TTLCache(ttl=0)
        c.set("k", 1)
        assert c.get("k") is None

    def test_lru_eviction(self):
        c = TTLCache(ttl=60, maxsize=2)
        c.set("a", 1); c.set("b", 2); c.set("c", 3)
        assert c.get("a") is None
        assert c.size == 2

    def test_get_or_load(self):
        c = TTLCache(ttl=60)
        calls = []

        def loader():
            calls.append(1)
            return "v"

        assert c.get_or_load("k", loader) == "v"
        assert c.get_or_load("k", loader) == "v"
        assert len(calls) == 1


class TestDiskCache:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "x.json"
        d = DiskCache(p, ttl=60)
        assert d.read_json() is None
        d.write_json({"a": 1})
        assert d.read_json()["a"] == 1
        assert d.fresh()
        d.invalidate()
        assert not p.exists()

    def test_corrupt_file_does_not_raise(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{ not json", encoding="utf-8")
        d = DiskCache(p, ttl=60)
        assert d.read_json() is None
        assert d.read_dataframe() is None

    def test_dataframe_roundtrip(self, tmp_path):
        d = DiskCache(tmp_path / "df.json", ttl=60)
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert d.write_dataframe(df)
        back = d.read_dataframe()
        assert list(back.columns) == ["a", "b"] and len(back) == 2


class TestMemoryCacheCompat:
    def test_ranking_ttl_zero_expires(self):
        mc = MemoryCache(ttl=0)
        mc.set_ranking(pd.DataFrame({"a": [1]}))
        assert mc.get_ranking() is None


class TestRateLimiter:
    def test_enforces_interval(self):
        rl = RateLimiter(min_interval=0.05)
        rl.acquire()
        waited = rl.acquire()
        assert waited > 0
        assert rl.waited == 1

    def test_zero_interval_is_noop(self):
        rl = RateLimiter(min_interval=0)
        assert rl.acquire() == 0.0


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        cb = CircuitBreaker(threshold=2, reset_timeout=60)
        cb.record_failure(); assert cb.state() == "closed"
        cb.record_failure(); assert cb.state() == "open"

    def test_success_resets(self):
        cb = CircuitBreaker(threshold=2, reset_timeout=60)
        cb.record_failure(); cb.record_success()
        assert cb.failures == 0 and cb.state() == "closed"

    def test_cooldown_closes_again(self):
        cb = CircuitBreaker(threshold=1, reset_timeout=0.01)
        cb.record_failure()
        assert cb.state() == "open"
        time.sleep(0.02)
        assert cb.state() == "closed"


class TestRetry:
    def test_retries_then_succeeds(self):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("boom")
            return "ok"

        out = call_with_retry(flaky, retries=3, backoff=0.001, label="test")
        assert out == "ok" and attempts["n"] == 3

    def test_raises_source_unavailable(self):
        def always_fail():
            raise RuntimeError("nope")

        with pytest.raises(SourceUnavailable):
            call_with_retry(always_fail, retries=1, backoff=0.001, label="test")

    def test_open_breaker_skips_call(self):
        cb = CircuitBreaker(threshold=1, reset_timeout=60)
        cb.record_failure()
        called = []
        with pytest.raises(SourceUnavailable):
            call_with_retry(lambda: called.append(1), breaker=cb, label="test")
        assert called == []


class TestRunBlocking:
    def test_runs_sync_fn_in_executor(self):
        import asyncio

        async def main():
            return await run_blocking(lambda x: x * 2, 21)

        assert asyncio.run(main()) == 42


class TestErrors:
    @pytest.mark.parametrize("exc_cls,status", [
        (InvalidInput, 400), (InsufficientData, 422), (SourceUnavailable, 503),
    ])
    def test_status_codes(self, exc_cls, status):
        e = exc_cls("坏了")
        assert e.status_code == status
        payload = e.to_payload()
        assert payload["message"] == "坏了"
        assert payload["detail"] == "坏了"   # 兼容旧前端读取 detail
        assert isinstance(e, AppError)


class TestBreakerRegistry:
    def test_keys_are_isolated(self):
        reg = BreakerRegistry(threshold=1, reset_timeout=60)
        reg.get("nav").record_failure()
        assert reg.state_for("nav") == "open"
        assert reg.state_for("ranking") == "closed"
        assert reg.open_keys() == ["nav"]
        assert reg.state() == "open"

    def test_unknown_key_reports_closed(self):
        reg = BreakerRegistry()
        assert reg.state_for("never-used") == "closed"

    def test_bounded_keys(self):
        reg = BreakerRegistry(max_keys=3)
        for i in range(10):
            reg.get(f"k{i}")
        assert len(reg.summary()) <= 3


class TestErrorClassification:
    def test_parse_errors_are_data_level(self):
        assert is_infrastructure_error(TypeError("no column")) is False
        assert is_infrastructure_error(KeyError("基金代码")) is False
        assert is_infrastructure_error(ValueError("empty")) is False

    def test_network_errors_are_infrastructure(self):
        class ConnectionError(Exception):
            pass

        class ReadTimeout(Exception):
            pass

        assert is_infrastructure_error(ConnectionError()) is True
        assert is_infrastructure_error(ReadTimeout()) is True


class TestDataFailureDoesNotOpenBreaker:
    """一只不存在的基金代码不能把整个净值接口的熔断打开。"""

    def test_breaker_stays_closed_on_data_level_failure(self):
        cb = CircuitBreaker(threshold=2, reset_timeout=60)

        def missing():
            raise TypeError("单元素子数组起的序列")

        for _ in range(5):
            with pytest.raises(SourceUnavailable):
                call_with_retry(missing, retries=2, backoff=0.001,
                                breaker=cb, label="净值 000002")
        assert cb.state() == "closed"

    def test_no_retry_on_data_level_failure(self):
        calls = []
        cb = CircuitBreaker(threshold=1, reset_timeout=60)

        def missing():
            calls.append(1)
            raise KeyError("单位净值")

        with pytest.raises(SourceUnavailable):
            call_with_retry(missing, retries=3, backoff=0.001, breaker=cb, label="净值")
        assert len(calls) == 1

    def test_infrastructure_failure_retries_and_opens(self):
        calls = []
        cb = CircuitBreaker(threshold=1, reset_timeout=60)

        class ConnectionError(Exception):
            pass

        def down():
            calls.append(1)
            raise ConnectionError("max retries exceeded")

        with pytest.raises(SourceUnavailable):
            call_with_retry(down, retries=2, backoff=0.001, breaker=cb, label="净值")
        assert len(calls) == 3
        assert cb.state() == "open"
