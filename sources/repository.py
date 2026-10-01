"""基金数据仓库：缓存编排 + 降级策略。

这里是唯一持有"数据状态"的地方（不再有模块级全局字典）。
一个请求仓库实例 = 一组缓存 + 一个 akshare 客户端，可以在测试里随意替换。

职责边界：仓库只返回原始数据（DataFrame / Series / dict），
不算指标、不拼文案 —— 那些属于 analysis 和 services。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import Settings
from core.cache import DiskCache, TTLCache
from core.errors import InsufficientData, InvalidInput, SourceUnavailable
from sources import normalize
from sources.akshare_client import AkshareClient

logger = logging.getLogger("fund-agent")


def slice_days(series: pd.Series, days: int) -> pd.Series:
    """按最近 N 天切片（days<=0 表示不切）。"""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float) if series is None else series
    if days and days > 0:
        cutoff = datetime.now() - timedelta(days=days)
        series = series[series.index >= pd.to_datetime(cutoff)]
    return series


class FundRepository:
    def __init__(self, settings: Settings, client: Optional[AkshareClient] = None):
        self.settings = settings
        cfg = settings.source
        self.client = client or AkshareClient(cfg)
        c = settings.cache
        self._ranking_mem: TTLCache[pd.DataFrame] = TTLCache(ttl=c.ranking_ttl, maxsize=4)
        self._detail_mem: TTLCache[dict] = TTLCache(ttl=c.detail_ttl, maxsize=c.max_detail_entries)
        self._nav_mem: TTLCache[pd.Series] = TTLCache(ttl=c.nav_ttl, maxsize=c.max_nav_entries)
        self._bench_mem: TTLCache[pd.Series] = TTLCache(ttl=c.nav_ttl, maxsize=8)
        self._news_mem: TTLCache[dict] = TTLCache(ttl=c.news_ttl, maxsize=4)
        self.ranking_disk = DiskCache(c.ranking_path, c.ranking_ttl)
        self.news_disk = DiskCache(c.news_path, c.news_ttl)

    # ========== 排名 ==========
    def ranking(self, force_refresh: bool = False) -> pd.DataFrame:
        """开放式基金排行表。内存 -> 磁盘 -> 网络 -> 过期磁盘降级。"""
        if not force_refresh:
            cached = self._ranking_mem.get("ranking")
            if cached is not None:
                return cached
            if self.ranking_disk.fresh():
                df = self.ranking_disk.read_dataframe()
                if df is not None and not df.empty:
                    logger.info(f"[文件缓存] 加载 {len(df)} 条基金数据")
                    age = self.ranking_disk.age() or 0.0
                    df = self._prepare(df)
                    self._ranking_mem.set("ranking", df, timestamp=time.time() - age)
                    return df

        try:
            df = self.client.ranking()
            logger.info(f"[抓取] 成功获取 {len(df)} 条基金数据")
            df = self._prepare(df)
            self.ranking_disk.write_dataframe(df)
            self._ranking_mem.set("ranking", df)
            return df
        except SourceUnavailable as e:
            stale = self.ranking_disk.read_dataframe()
            if stale is not None and not stale.empty:
                logger.warning(f"[降级] 外部数据不可用（{e}），使用旧缓存 {len(stale)} 条")
                self._ranking_mem.set("ranking", self._prepare(stale))
                return self._ranking_mem.get("ranking")
            raise SourceUnavailable(f"基金数据不可用且无本地缓存: {e}")

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        """出仓前的统一清洗：目前只有代码零补齐（缓存往返会丢前导零）。"""
        return normalize.normalize_codes(df)

    def ranking_meta(self) -> dict:
        return {
            "memory": self._ranking_mem.stats(),
            "disk_age_seconds": self.ranking_disk.age(),
            "degraded": self.client.breakers.state_for("fund_open_fund_rank_em"),
        }

    def ranking_cached(self) -> bool:
        """内存里是否已有可用的排名数据。"""
        return self._ranking_mem.get("ranking") is not None

    def detail_cache_size(self) -> int:
        return self._detail_mem.size

    def nav_source_blocked(self) -> bool:
        """净值接口是否正处于熔断冷却。用来区分"这只基金不存在"和"上游在限流"。"""
        return self.client.breakers.state_for("fund_open_fund_info_em") == "open"

    # ========== 列名解析 ==========
    def resolve(self, df: pd.DataFrame) -> Dict[str, str]:
        return normalize.resolve_columns(df)

    # ========== 净值 / 基准 ==========
    def nav_history(self, code: str, days: int = 365) -> pd.Series:
        if not str(code).isdigit() or len(str(code)) != 6:
            raise InvalidInput(f"基金代码格式错误: {code}")
        cached = self._nav_mem.get(code)
        if cached is not None:
            return slice_days(cached, days)
        try:
            series = self.client.nav_history(code)
        except SourceUnavailable as e:
            logger.warning(f"[净值] {code} 获取失败: {e}")
            return pd.Series(dtype=float)
        if len(series) > 1:
            self._nav_mem.set(code, series)
        return slice_days(series, days)

    def benchmark(self, days: int = 365, symbol: Optional[str] = None) -> pd.Series:
        sym = symbol or self.settings.source.benchmark_symbol
        cached = self._bench_mem.get(sym)
        if cached is not None:
            return slice_days(cached, days)
        try:
            series = self.client.benchmark(sym)
        except SourceUnavailable as e:
            logger.warning(f"[基准] {sym} 获取失败: {e}")
            return pd.Series(dtype=float)
        if len(series) > 1:
            self._bench_mem.set(sym, series)
        return slice_days(series, days)

    def aligned_returns(self, codes: List[str], days: int = 365
                        ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        多只基金对齐后的日收益率（仅保留所有列都有值的公共交易日）+ 各自净值切片。

        返回 (returns_df, nav_by_code)。数据不足时 returns_df 为空，由上层决定报错文案。
        """
        nav_by_code: Dict[str, pd.Series] = {}
        for code in codes:
            s = self.nav_history(code, days=days)
            if len(s) > 1:
                nav_by_code[code] = s
        if not nav_by_code:
            return pd.DataFrame(), nav_by_code

        frame = pd.DataFrame({c: pd.Series(s.values, index=s.index) for c, s in nav_by_code.items()})
        returns = frame.pct_change().dropna(how="any")
        return returns, nav_by_code

    # ========== 基本信息（详情用） ==========
    def basic_info(self, code: str) -> dict:
        cached = self._detail_mem.get(f"info:{code}")
        if cached is not None:
            return cached
        info = self.client.basic_info(code)
        if info:
            self._detail_mem.set(f"info:{code}", info)
        return info

    def cache_detail(self, code: str, detail: dict) -> None:
        self._detail_mem.set(f"detail:{code}", detail)

    def cached_detail(self, code: str) -> Optional[dict]:
        return self._detail_mem.get(f"detail:{code}")

    # ========== 资讯 ==========
    def news_payload(self) -> dict:
        """返回 {"news": [...], "boards": [...]}，失败返回空结构（资讯不是关键路径）。"""
        cached = self._news_mem.get("news")
        if cached is not None:
            return cached
        if self.news_disk.fresh():
            disk = self.news_disk.read_json()
            if disk:
                self._news_mem.set("news", disk)
                return disk
        payload = {"news": [], "boards": [], "ts": None}
        try:
            payload["news"] = self.client.news()
        except SourceUnavailable as e:
            logger.warning(f"[资讯] 快讯抓取失败: {e}")
        try:
            payload["boards"] = self.client.boards()
        except SourceUnavailable as e:
            logger.warning(f"[资讯] 板块抓取失败: {e}")
        if payload["news"] or payload["boards"]:
            payload["ts"] = datetime.now().isoformat()
            self.news_disk.write_json(payload)
            self._news_mem.set("news", payload)
        return payload

    # ========== 缓存管理 ==========
    def invalidate_all(self) -> None:
        for cache in (self._ranking_mem, self._detail_mem, self._nav_mem,
                      self._bench_mem, self._news_mem):
            cache.clear()
        self.ranking_disk.invalidate()
        self.news_disk.invalidate()

    def health(self) -> dict:
        return {
            "ranking": self._ranking_mem.stats(),
            "nav": self._nav_mem.stats(),
            "detail": self._detail_mem.stats(),
            "disk_age_seconds": self.ranking_disk.age(),
            "source": self.client.stats(),
        }

    def require_aligned(self, codes: List[str], days: int, min_funds: int = 2
                        ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """对齐数据 + 样本量校验（不足则抛 InsufficientData，并说清楚缺在哪）。"""
        returns, nav_by_code = self.aligned_returns(codes, days=days)
        missing = [c for c in codes if c not in nav_by_code]
        if returns.shape[1] < min_funds:
            hint = f"；其中 {len(missing)} 只没取到净值（{', '.join(missing[:5])}），" \
                   f"可能是代码不存在或上游暂时不可用" if missing else ""
            raise InsufficientData(
                f"有效基金仅 {returns.shape[1]} 只（需要 {min_funds} 只以上）{hint}，"
                f"请更换代码或延长时间区间")
        cfg = self.settings.analysis
        if len(returns) < cfg.min_sample_days:
            raise InsufficientData(
                f"对齐后样本仅 {len(returns)} 个交易日，建议不少于 {cfg.min_sample_days} 天")
        return returns, nav_by_code
