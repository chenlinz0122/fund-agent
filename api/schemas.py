"""请求/响应模型（pydantic）。

与 services 层的 dataclass 条件对象（FundCriteria）分开：
HTTP 形状变化不应该波及计算逻辑。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, constr

FundCode = constr(pattern=r"^\d{6}$")


class ChatRequest(BaseModel):
    message: str
    history: list = []
    use_llm: bool = Field(True, description="False 时强制走规则引擎，便于对比与省 token")


class ScreenRequest(BaseModel):
    fund_type: Optional[str] = None
    min_return_1y: Optional[float] = None
    max_return_1y: Optional[float] = None
    min_sharpe: Optional[float] = None
    min_size: Optional[float] = None
    max_size: Optional[float] = None
    sort_by: str = "return_1y"
    sort_order: str = "desc"
    limit: int = Field(50, ge=1, le=500, description="返回数量，最多500条")

    def to_criteria(self):
        from services.screening import FundCriteria
        return FundCriteria(
            fund_type=self.fund_type, min_return_1y=self.min_return_1y,
            max_return_1y=self.max_return_1y, min_sharpe=self.min_sharpe,
            min_size=self.min_size, max_size=self.max_size,
            sort_by=self.sort_by, sort_order=self.sort_order, limit=self.limit,
        )


class PortfolioRequest(BaseModel):
    codes: List[str] = Field(..., min_length=2, max_length=12)
    days: int = Field(365, ge=30, le=2000)


class BacktestRequest(PortfolioRequest):
    weights: Optional[Dict[str, float]] = None
    rebalance: str = Field("monthly", pattern="^(none|monthly|daily)$")
    cost_pct: Optional[float] = Field(None, ge=0, le=5,
                                      description="单边换手成本（百分数）。用于估算频繁调仓的费用侵蚀")


class RobustnessRequest(BaseModel):
    codes: List[str] = Field(..., min_length=3, max_length=12)
    days: int = Field(1095, ge=365, le=3000, description="需要的历史长度")
    train_days: int = Field(252, ge=60, le=756)
    test_days: int = Field(63, ge=21, le=252)
    folds: int = Field(4, ge=2, le=12)
    n_trials: int = Field(1, ge=1, le=10000,
                          description="你为这批参数尝试过的变体数量，用于 Deflated Sharpe 惩罚")


class ExpectancyRequest(BaseModel):
    codes: List[str] = Field(..., min_length=1, max_length=12)
    days: int = Field(365, ge=30, le=2000)
    round_trips_per_year: Optional[float] = Field(None, ge=0, le=250,
                                                  description="预计每年申赎往返次数")
    fund_types: Optional[Dict[str, str]] = None


class SetModelRequest(BaseModel):
    model: str


class SetBudgetRequest(BaseModel):
    budget_tokens: Optional[int] = None


class SimulateRequest(BaseModel):
    codes: List[str] = Field(..., min_length=2, max_length=12)
    days: int = Field(365, ge=90, le=2000)
    season: Optional[str] = Field(None, description="季节层资金激进乘数：冬/春/夏/秋")
    weights: Optional[Dict[str, float]] = None
    genes: Optional[Dict[str, Any]] = Field(None, description="可进化基因覆盖；环境层字段会被忽略")
    exit: Optional[Dict[str, Any]] = None


class EvolveRequest(BaseModel):
    codes: List[str] = Field(..., min_length=2, max_length=8)
    days: int = Field(500, ge=180, le=2000)
    pop_size: int = Field(16, ge=6, le=40)
    generations: int = Field(4, ge=1, le=12)
    seed: int = Field(2026, ge=1, le=999999)
    incumbent: Optional[Dict[str, Any]] = Field(None, description="当前冠军基因，作为 10% 火种注入")
    exit: Optional[Dict[str, Any]] = None
    plateau: bool = Field(True, description="是否顺带做参数平原检验（多花约 13 次评估）")


class PolicyRequest(BaseModel):
    codes: List[str] = Field(..., min_length=1, max_length=12)
    days: int = Field(365, ge=90, le=2000)
    genes: Optional[Dict[str, Any]] = None


# ========== 模拟盘 ==========
class OpenAccountRequest(BaseModel):
    name: str = Field("模拟盘", max_length=40)
    capital: float = Field(100000, gt=0, le=1e9)
    codes: List[str] = Field(default_factory=list, max_length=12)
    strategy_id: Optional[str] = None
    seed_buy: bool = Field(False, description="建仓：把初始资金等权申购进 codes")


class OrderRequest(BaseModel):
    code: str
    side: str = Field(..., pattern="^(buy|sell)$")
    amount: Optional[float] = Field(None, gt=0, description="申购金额（元）")
    shares: Optional[float] = Field(None, gt=0, description="赎回份额；省略表示全部")
    reason: str = Field("", max_length=200)


# ========== 自训练 ==========
class TrainRoundRequest(BaseModel):
    codes: List[str] = Field(..., min_length=2, max_length=8)
    days: int = Field(1200, ge=400, le=2500)
    pop_size: int = Field(14, ge=6, le=40)
    generations: int = Field(3, ge=1, le=12)
    seed: int = Field(2026, ge=1, le=999999)
    holdout_days: int = Field(60, ge=20, le=252, description="从进化过程中藏起来的留出段长度")


class PromoteRequest(BaseModel):
    record_id: str
    note: str = Field("", max_length=200)
