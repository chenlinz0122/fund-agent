"""分析层常量。

只在这里定义一次，避免 252 / 0.02 这类数字散落在各处。
需要不同取值的调用方请显式传参（risk_free=...），不要改这里。
"""
from __future__ import annotations

TRADING_DAYS = 252
DEFAULT_RISK_FREE = 0.02      # 年化无风险利率 ~2%
EPS = 1e-8

# 兼容旧命名
_EPS = EPS
