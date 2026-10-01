"""意图路由。

原 rule_based_agent 是一条 90 行的 if 链：关键词散落、优先级靠代码顺序隐式决定，
加一个新意图就得读懂整条链。现在关键词与优先级集中在 INTENTS 表里，
处理函数只按名字注册，互不影响。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

FUND_TYPES = ["股票", "债券", "混合", "指数", "货币", "QDII", "量化"]


@dataclass(frozen=True)
class Intent:
    name: str
    keywords: tuple
    description: str
    needs_data: bool = True


# 顺序即优先级：市场资讯优先，避免被"收益/行情"等词抢走；概念题其次走知识库。
INTENTS: List[Intent] = [
    Intent("market", ("新闻", "热点", "资讯", "快讯", "板块", "行情", "市场动向",
                      "今天市场", "大盘"), "市场资讯与板块异动"),
    Intent("concept", ("什么是", "是什么", "怎么", "如何", "为什么", "什么意思",
                       "科普", "定义", "怎样", "区别"), "概念科普（知识库）", needs_data=False),
    Intent("fund_type_overview", ("基金类型", "有哪些类型", "什么类型", "分类",
                                  "哪些种类", "怎么分类"), "基金类型总览", needs_data=False),
    Intent("recommend", ("推荐", "建议", "买什么", "哪个好", "有没有好的", "挑选"), "基金推荐"),
    Intent("compare", ("对比", "比较", "差别"), "多基金对比"),
    Intent("strategy", ("仓位", "止损", "止盈", "策略", "调仓", "再平衡", "信号",
                        "加杠杆", "做空", "遗传", "进化", "参数", "定投"),
           "仓位策略与止盈止损"),
    Intent("risk", ("风险", "稳健", "安全", "回撤", "保守", "低风险"), "稳健/风险筛选"),
    Intent("expectancy", ("费用", "手续费", "赎回费", "管理费", "费后", "净值损耗",
                          "期望", "划不划算", "值不值"), "费后期望体检"),
    Intent("ranking", ("排行", "排名", "最高", "最好", "top", "榜"), "收益排行榜"),
]

INTENT_BY_NAME: Dict[str, Intent] = {i.name: i for i in INTENTS}


def extract_codes(message: str, limit: int = 3) -> List[str]:
    """提取 6 位基金代码。"""
    return re.findall(r"\d{6}", message or "")[:limit]


def extract_return_threshold(message: str) -> Optional[float]:
    """从消息中提取百分比收益阈值（如 20% -> 20.0）。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", message or "")
    return float(m.group(1)) if m else None


def extract_fund_type_from_message(message: str) -> Optional[str]:
    """识别基金类型关键词。"""
    low = (message or "").lower()
    for t in FUND_TYPES:
        if t.lower() in low:
            return t
    return None


def extract_round_trips(message: str) -> Optional[float]:
    """提取年换手次数（"一年买卖3次" / "每年调仓 4 次"）。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:次|回)", message or "")
    if m:
        return float(m.group(1))
    return None


class IntentRouter:
    """按优先级返回候选意图；处理函数可以返回 None 表示"我答不了，交给下一个"。"""

    def __init__(self, intents: Optional[List[Intent]] = None):
        self.intents = intents or INTENTS

    def match(self, message: str) -> List[str]:
        low = (message or "").lower()
        return [i.name for i in self.intents
                if any(k.lower() in low for k in i.keywords)]

    def route(self, message: str) -> Optional[str]:
        names = self.match(message)
        return names[0] if names else None

    def describe(self, message: str) -> dict:
        """用于调试与前端提示：这条消息被识别成了什么意图。"""
        names = self.match(message)
        return {
            "intent": names[0] if names else "fallback",
            "candidates": names,
            "codes": extract_codes(message, limit=6),
            "fund_type": extract_fund_type_from_message(message),
            "return_threshold": extract_return_threshold(message),
        }
