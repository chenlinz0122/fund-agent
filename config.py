"""
全局配置 —— 项目里唯一读取环境变量的地方。

设计约定（见 ARCHITECTURE.md）：
  - 其它模块只接收 Settings 实例，不自己 os.getenv()，避免配置散落各处。
  - Settings 是 frozen 的：运行期不可变，防止出现隐藏的全局状态。
  - 运行期可变的量（当前模型、token 用量）放在 agent.llm 里，不混进配置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv 缺失时仍可运行（例如纯离线分析）
    def load_dotenv(*_a, **_kw):
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"
KNOWLEDGE_FILE = DATA_DIR / "knowledge_base.json"


def _env_str(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# ========== 分层配置 ==========
@dataclass(frozen=True)
class CacheConfig:
    """各类数据的缓存时长（秒）与落盘位置。"""
    ranking_ttl: int = 3600 * 4      # 基金排名 4 小时
    detail_ttl: int = 3600           # 基金详情 1 小时
    nav_ttl: int = 1800              # 净值历史 30 分钟
    news_ttl: int = 1800             # 资讯 30 分钟
    max_nav_entries: int = 256       # 净值缓存条目上限，防止长时间运行后内存无界增长
    max_detail_entries: int = 512
    ranking_file: str = "fund_ranking_cache.json"
    news_file: str = "news_cache.json"

    def path(self, name: str) -> Path:
        return DATA_DIR / name

    @property
    def ranking_path(self) -> Path:
        return self.path(self.ranking_file)

    @property
    def news_path(self) -> Path:
        return self.path(self.news_file)


@dataclass(frozen=True)
class SourceConfig:
    """外部数据源（akshare）的韧性参数。

    免费公开接口有速率限制，所以这里默认是"温和请求"：
    限流 -> 重试 -> 熔断降级到缓存，而不是把上游打爆。
    """
    min_interval: float = 0.25       # 两次外部调用之间的最小间隔（秒）
    max_retries: int = 2             # 失败重试次数（不含首次）
    backoff: float = 0.6             # 退避基数：第 n 次重试等待 backoff * 2**n
    timeout: float = 20.0
    breaker_threshold: int = 5       # 连续失败多少次后熔断（免费接口偶发超时很常见，别太敏感）
    breaker_reset: float = 45.0      # 熔断打开后的冷却时间（秒）
    fund_symbol: str = "全部"         # akshare 排名接口品种
    benchmark_symbol: str = "sh000300"
    benchmark_name: str = "沪深300"


@dataclass(frozen=True)
class AnalysisConfig:
    """量化分析参数。"""
    trading_days: int = 252
    risk_free: float = 0.02          # 年化无风险利率
    min_sample_days: int = 30        # 组合优化/回测所需最少对齐样本
    corr_warn_threshold: float = 0.85
    target_volatility: float = 0.10  # 组合目标年化波动率（不加杠杆）
    max_gross_exposure: float = 1.0  # 硬约束：总仓位 <= 100%，即禁止杠杆
    corr_lookback_for_vol: int = 63  # 估计已实现波动率的窗口（约一个季度）
    luck_concentration_warn: float = 0.5  # 前 5 个交易日贡献超过一半收益时预警


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = ""
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-turbo"
    timeout: float = 30.0
    temperature: float = 0.7
    max_tokens: int = 1000
    history_limit: int = 5
    context_chars: int = 4000        # 注入给模型的上下文长度上限
    budget_tokens: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


# ========== 费用假设（残酷数学真相：费用是确定性负期望） ==========
# A 股公募基金常见费率。默认按"主流互联网代销平台打 1 折后的申购费"取值。
# 单位：百分比。holding_years 指平均持有时长，用于摊薄管理费。
DEFAULT_FEE_PROFILES: Dict[str, Dict[str, float]] = {
    # subscription: 申购费(折后)  redemption: 赎回费(持有>=1年通常为0)
    # management / custody:  annual_pct  每年从净值中计提
    "股票型": {"subscription": 0.15, "redemption": 0.50, "management": 1.20, "custody": 0.20},
    "混合型": {"subscription": 0.15, "redemption": 0.50, "management": 1.20, "custody": 0.20},
    "指数型": {"subscription": 0.10, "redemption": 0.50, "management": 0.50, "custody": 0.10},
    "QDII":  {"subscription": 0.12, "redemption": 0.50, "management": 1.00, "custody": 0.20},
    "债券型": {"subscription": 0.08, "redemption": 0.10, "management": 0.60, "custody": 0.10},
    "货币型": {"subscription": 0.00, "redemption": 0.00, "management": 0.33, "custody": 0.05},
}
DEFAULT_FEE_TYPE = "混合型"

# 持有不足 7 天的惩罚性赎回费（监管规定，不低于 1.5%）。
SHORT_TERM_PENALTY_REDEMPTION = {
    "days": 7,
    "redemption": 1.50,
}


@dataclass(frozen=True)
class FeeConfig:
    """费率假设。真实费率以基金法律文件为准，这里只用于估计费用侵蚀量级。"""
    profiles: Dict[str, Dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_FEE_PROFILES))
    default_type: str = DEFAULT_FEE_TYPE
    penalty_days: int = SHORT_TERM_PENALTY_REDEMPTION["days"]
    penalty_redemption: float = SHORT_TERM_PENALTY_REDEMPTION["redemption"]
    default_holding_years: float = 1.0
    default_round_trips_per_year: float = 1.0
    # 赎回费阶梯：(持有天数上限, 费率%)，按"确认日 - 份额买入日"落在哪一档取值。
    # 第一档就是监管要求的惩罚性赎回费（<7 天不低于 1.5%）。
    redemption_tiers: tuple = ((7, 1.50), (30, 0.75), (365, 0.50), (None, 0.00))

    def redemption_rate(self, holding_days: float) -> float:
        """按持有天数给出赎回费率（百分数）。"""
        for limit, rate in self.redemption_tiers:
            if limit is None or holding_days < limit:
                return float(rate)
        return 0.0

    def subscription_rate(self, fund_type: Optional[str] = None) -> float:
        profile = self.profiles.get(_norm_type(fund_type)) or self.profiles[self.default_type]
        return float(profile.get("subscription", 0.0))

    def base_redemption_rate(self, fund_type: Optional[str] = None) -> float:
        profile = self.profiles.get(_norm_type(fund_type)) or self.profiles[self.default_type]
        return float(profile.get("redemption", 0.0))


def _norm_type(fund_type: Optional[str]) -> str:
    if not fund_type:
        return ""
    t = str(fund_type).strip()
    if t in {"股票", "混合", "债券", "指数", "货币", "QDII"}:
        t += "型"
    return t


@dataclass(frozen=True)
class Settings:
    """应用配置聚合根。"""
    cache: CacheConfig = field(default_factory=CacheConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    factor_weights: Optional[Dict[str, float]] = None   # None -> 用 analysis.factors 默认权重
    data_dir: Path = DATA_DIR

    @property
    def knowledge_file(self) -> Path:
        return KNOWLEDGE_FILE

    def to_dict(self) -> dict:
        return {
            "llm_enabled": self.llm.enabled,
            "llm_model": self.llm.model,
            "cache": {
                "ranking_ttl": self.cache.ranking_ttl,
                "detail_ttl": self.cache.detail_ttl,
                "nav_ttl": self.cache.nav_ttl,
                "news_ttl": self.cache.news_ttl,
            },
            "source": {
                "min_interval": self.source.min_interval,
                "max_retries": self.source.max_retries,
                "breaker_threshold": self.source.breaker_threshold,
            },
            "analysis": {
                "risk_free": self.analysis.risk_free,
                "target_volatility": self.analysis.target_volatility,
                "max_gross_exposure": self.analysis.max_gross_exposure,
            },
        }


def load_settings() -> Settings:
    """从 .env / 环境变量构造 Settings（唯一入口）。"""
    load_dotenv()

    origins_raw = _env_str("ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()] or ["*"]

    budget = _env_int("LLM_BUDGET_TOKENS", 0)

    cache = CacheConfig(
        ranking_ttl=_env_int("CACHE_RANKING_TTL", 3600 * 4),
        detail_ttl=_env_int("CACHE_DETAIL_TTL", 3600),
        nav_ttl=_env_int("CACHE_NAV_TTL", 1800),
        news_ttl=_env_int("CACHE_NEWS_TTL", 1800),
    )
    source = SourceConfig(
        min_interval=_env_float("SOURCE_MIN_INTERVAL", 0.25),
        max_retries=_env_int("SOURCE_MAX_RETRIES", 2),
        breaker_threshold=_env_int("SOURCE_BREAKER_THRESHOLD", 3),
        breaker_reset=_env_float("SOURCE_BREAKER_RESET", 120.0),
    )
    analysis = AnalysisConfig(
        risk_free=_env_float("ANNUAL_RISK_FREE", 0.02),
        target_volatility=_env_float("TARGET_VOLATILITY", 0.10),
    )
    llm = LLMConfig(
        api_key=_env_str("LLM_API_KEY"),
        api_base=_env_str("LLM_API_BASE", LLMConfig.api_base),
        model=_env_str("LLM_MODEL", LLMConfig.model),
        timeout=_env_float("LLM_TIMEOUT", 30.0),
        temperature=_env_float("LLM_TEMPERATURE", 0.7),
        max_tokens=_env_int("LLM_MAX_TOKENS", 1000),
        budget_tokens=budget if budget > 0 else None,
    )
    DATA_DIR.mkdir(exist_ok=True)

    return Settings(
        cache=cache, source=source, analysis=analysis, llm=llm,
        fees=FeeConfig(), allow_origins=origins, data_dir=DATA_DIR,
    )
