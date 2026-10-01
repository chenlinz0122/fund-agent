"""基金筛选用例。

关键改进：原实现里 ScreenRequest 的 min_sharpe / min_size / max_size 三个条件
被静默忽略 —— 用户勾了筛选框却什么都没发生。现在每个条件要么真的生效，
要么明确报告"该字段当前数据源没有，已忽略"。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from sources import normalize

# 额外可映射的筛选字段（不同 akshare 版本列名不同，这里统一容错）
EXTRA_MAPPING: Dict[str, List[str]] = {
    "sharpe": ["Sharpe", "夏普比率", "Sharpe比率", "波动率"],
    "size": ["基金规模(亿元)", "基金规模", "规模亿元", "净资产规模(亿元)"],
}


@dataclass
class FundCriteria:
    """筛选条件（与 HTTP 层解耦，方便单测和复用）。"""
    fund_type: Optional[str] = None
    min_return_1y: Optional[float] = None
    max_return_1y: Optional[float] = None
    min_sharpe: Optional[float] = None
    min_size: Optional[float] = None
    max_size: Optional[float] = None
    sort_by: str = "return_1y"
    sort_order: str = "desc"
    limit: int = 50


@dataclass
class ScreenResult:
    funds: List[dict]
    applied: List[str]
    ignored: List[str]
    total_before_filter: int

    def to_dict(self) -> dict:
        return {
            "funds": self.funds,
            "count": len(self.funds),
            "total_before_filter": self.total_before_filter,
            "applied_filters": self.applied,
            "ignored_filters": self.ignored,
            "ignored_reason": ("以下条件在当前数据源中不存在对应字段，已忽略"
                               if self.ignored else None),
        }


def _resolve_extra(df: pd.DataFrame) -> Dict[str, str]:
    """在 COLUMN_MAPPING 之外补充规模/夏普类列。"""
    out = {}
    cols = list(df.columns)
    for key, candidates in EXTRA_MAPPING.items():
        for cand in candidates:
            if cand in cols:
                out[key] = cand
                break
        else:
            for col in cols:
                if any(cand in str(col) for cand in candidates):
                    out[key] = col
                    break
    return out


def apply_criteria(df: pd.DataFrame, criteria: FundCriteria) -> ScreenResult:
    """按条件筛选并排序，返回结果 + 条件生效情况。"""
    result = df.copy()
    col_map = normalize.resolve_columns(result)
    col_map.update(_resolve_extra(result))
    normalize.coerce_numeric(result, col_map)
    applied: List[str] = []
    ignored: List[str] = []

    if criteria.fund_type:
        ft = str(criteria.fund_type).strip()
        ft_base = ft[:-1] if ft.endswith("型") else ft
        if "type" in col_map:
            mask = result[col_map["type"]].astype(str).str.contains(ft, na=False, regex=False)
            if "name" in col_map:
                mask |= result[col_map["name"]].astype(str).str.contains(ft_base, na=False, regex=False)
            result = result[mask]
            applied.append(f"fund_type={ft}")
        elif "name" in col_map:
            result = result[result[col_map["name"]].astype(str).str.contains(ft_base, na=False, regex=False)]
            applied.append(f"fund_type={ft}(按名称匹配)")
        else:
            ignored.append("fund_type")

    pairs = [
        ("min_return_1y", "return_1y", "ge", criteria.min_return_1y),
        ("max_return_1y", "return_1y", "le", criteria.max_return_1y),
        ("min_sharpe", "sharpe", "ge", criteria.min_sharpe),
        ("min_size", "size", "ge", criteria.min_size),
        ("max_size", "size", "le", criteria.max_size),
    ]
    for name, alias, op, value in pairs:
        if value is None:
            continue
        col = col_map.get(alias)
        if col is None or col not in result.columns:
            ignored.append(name)
            continue
        series = pd.to_numeric(result[col], errors="coerce")
        result = result[series >= value] if op == "ge" else result[series <= value]
        applied.append(f"{name}={value}")

    sort_col = col_map.get(criteria.sort_by) or col_map.get("return_1y")
    if sort_col and sort_col in result.columns:
        result = result.sort_values(
            sort_col, ascending=(criteria.sort_order == "asc"), na_position="last")
        applied.append(f"sort_by={criteria.sort_by} {criteria.sort_order}")
    else:
        ignored.append(f"sort_by={criteria.sort_by}")

    result = result.head(max(int(criteria.limit), 0))
    funds = normalize.to_records(result, col_map)
    return ScreenResult(
        funds=funds, applied=applied, ignored=ignored,
        total_before_filter=int(len(df)),
    )


def analyze_funds(df: pd.DataFrame, request) -> List[dict]:
    """
    兼容旧调用：request 可以是 FundCriteria，也可以是带同名属性的 pydantic 模型。
    只返回记录列表（原 main.analyze_funds 的行为）。
    """
    criteria = FundCriteria(
        fund_type=getattr(request, "fund_type", None),
        min_return_1y=getattr(request, "min_return_1y", None),
        max_return_1y=getattr(request, "max_return_1y", None),
        min_sharpe=getattr(request, "min_sharpe", None),
        min_size=getattr(request, "min_size", None),
        max_size=getattr(request, "max_size", None),
        sort_by=getattr(request, "sort_by", "return_1y") or "return_1y",
        sort_order=getattr(request, "sort_order", "desc") or "desc",
        limit=int(getattr(request, "limit", 50) or 50),
    )
    return apply_criteria(df, criteria).funds


def to_csv(funds: List[dict]) -> str:
    """导出 CSV（带 BOM，Excel 中文不乱码）。"""
    all_cols: List[str] = []
    for f in funds:
        for c in f.keys():
            if c not in all_cols:
                all_cols.append(c)
    preferred = normalize.PREFERRED_EXPORT_COLS
    ordered = [c for c in preferred if c in all_cols] + [c for c in all_cols if c not in preferred]

    buf = ["﻿"]
    if ordered:
        buf.append(",".join(csv_escape(c) for c in ordered) + "\n")
        for f in funds:
            buf.append(",".join(csv_escape(f.get(c)) for c in ordered) + "\n")
    return "".join(buf)


def csv_escape(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if any(ch in s for ch in [",", '"', "\n"]):
        s = '"' + s.replace('"', '""') + '"'
    return s
