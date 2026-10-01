"""akshare 客户端 —— 全项目唯一 import akshare 的地方。

每个方法 = 一次外部调用，统一经过 限流 -> 重试 -> 熔断。
调用失败抛 SourceUnavailable，由 repository 决定是否降级到缓存。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from config import SourceConfig
from core.errors import SourceUnavailable
from core.resilience import BreakerRegistry, RateLimiter, call_with_retry
from sources.normalize import pick_column, safe_float

logger = logging.getLogger("fund-agent")

_ak_module: Optional[Any] = None


def akshare_module():
    """惰性导入 akshare：没装或导入失败时不影响其它模块（分析层仍可离线使用）。"""
    global _ak_module
    if _ak_module is None:
        try:
            import akshare as ak  # noqa: F401
            _ak_module = ak
        except Exception as e:
            raise SourceUnavailable(f"akshare 不可用: {e}")
    return _ak_module


class AkshareClient:
    def __init__(self, cfg: Optional[SourceConfig] = None,
                 limiter: Optional[RateLimiter] = None,
                 breakers: Optional[BreakerRegistry] = None):
        self.cfg = cfg or SourceConfig()
        self.limiter = limiter or RateLimiter(self.cfg.min_interval)
        # 熔断按"接口"而不是按"整个应用"计数：一只不存在的基金不该封掉全部净值查询
        self.breakers = breakers or BreakerRegistry(self.cfg.breaker_threshold,
                                                    self.cfg.breaker_reset)
        self.calls = 0

    # ---- 内部统一调用 ----
    def _run(self, label: str, fn_name: str, *args, **kwargs):
        ak = akshare_module()
        fn = getattr(ak, fn_name, None)
        if fn is None:
            raise SourceUnavailable(f"当前 akshare 版本没有 {fn_name} 接口")
        self.calls += 1
        return call_with_retry(
            fn, *args,
            retries=self.cfg.max_retries, backoff=self.cfg.backoff,
            limiter=self.limiter, breaker=self.breakers.get(fn_name), label=label, **kwargs,
        )

    # ---- 原始抓取 ----
    def ranking(self) -> pd.DataFrame:
        """开放式基金排行（默认"全部"品种）。"""
        df = self._run("基金排行", "fund_open_fund_rank_em", symbol=self.cfg.fund_symbol)
        if df is None or (hasattr(df, "empty") and df.empty):
            raise SourceUnavailable("基金排行返回空数据")
        return df

    def basic_info(self, code: str) -> dict:
        """基金基本信息（字段->值）。"""
        try:
            info_df = self._run(f"基本信息 {code}", "fund_individual_basic_info_xq", symbol=code)
        except SourceUnavailable as e:
            logger.warning(f"[基本信息] {code} 获取失败: {e}")
            return {}
        if info_df is None or info_df.empty or info_df.shape[1] < 2:
            return {}
        return dict(zip(info_df.iloc[:, 0].astype(str), info_df.iloc[:, 1]))

    def nav_history(self, code: str) -> pd.Series:
        """单位净值走势 -> 以日期为索引的净值 Series（升序、去重、去空）。"""
        df = self._run(f"净值 {code}", "fund_open_fund_info_em", symbol=code, indicator="单位净值走势")
        if df is None or len(df) == 0 or df.shape[1] < 2:
            return pd.Series(dtype=float)
        date_col = pick_column(df, 0, "净值日期")
        nav_col = pick_column(df, 1, "单位净值")
        values = pd.to_numeric(df[nav_col], errors="coerce").values
        index = pd.to_datetime(df[date_col], errors="coerce")
        s = pd.Series(values, index=index, dtype=float).dropna()
        return s[~s.index.duplicated(keep="last")].sort_index()

    def benchmark(self, symbol: Optional[str] = None) -> pd.Series:
        """基准指数收盘价序列。"""
        sym = symbol or self.cfg.benchmark_symbol
        df = self._run(f"基准 {sym}", "stock_zh_index_daily", symbol=sym)
        if df is None or len(df) == 0:
            return pd.Series(dtype=float)
        frame = df.copy()
        date_col = "date" if "date" in frame.columns else frame.columns[0]
        close_col = "close" if "close" in frame.columns else frame.columns[-1]
        index = pd.to_datetime(frame[date_col], errors="coerce")
        s = pd.Series(pd.to_numeric(frame[close_col], errors="coerce").values, index=index, dtype=float).dropna()
        return s[~s.index.duplicated(keep="last")].sort_index()

    def news(self) -> list:
        """财联社快讯。"""
        df = self._run("财经快讯", "stock_news_main_cx")
        items = []
        if df is None or df.empty:
            return items
        for _, row in df.iterrows():
            items.append({
                "tag": str(row.get("tag", "")),
                "summary": str(row.get("summary", "")).strip(),
                "url": str(row.get("url", "")),
            })
        return items

    def boards(self) -> list:
        """板块异动（涨跌幅）。"""
        df = self._run("板块异动", "stock_board_change_em")
        boards = []
        if df is None or df.empty:
            return boards
        for _, row in df.iterrows():
            boards.append({
                "name": str(row.iloc[0]),
                "change_pct": safe_float(row.iloc[1]) if len(row) > 1 else None,
            })
        return boards

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "breaker": self.breakers.state(),
            "open_breakers": self.breakers.open_keys(),
            "breakers": self.breakers.summary(),
            "limiter": self.limiter.stats(),
        }
