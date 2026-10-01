"""列名归一化。

akshare 返回的中文列名会随版本变化（"近1月" vs "近一月"，3 列 vs 4 列）。
所有这类不确定性只在这一层吸收，上层只认标准字段名。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("fund-agent")

# 标准字段名 -> 候选实际列名（按优先级）
COLUMN_MAPPING: Dict[str, List[str]] = {
    "code": ["基金代码", "代码", "fund_code"],
    "name": ["基金简称", "简称", "基金名称", "名称", "fund_name"],
    "type": ["基金类型", "类型", "fund_type"],
    "return_1w": ["近1周", "近一周"],
    "return_1m": ["近1月", "近一月", "近1月收益"],
    "return_3m": ["近3月", "近三月", "近3个月", "近3月收益"],
    "return_6m": ["近6月", "近六月", "近半年", "近6月收益"],
    "return_1y": ["近1年", "近一年", "近1年收益"],
    "return_2y": ["近2年", "近两年"],
    "return_3y": ["近3年", "近三年"],
    "return_ytd": ["今年来", "今年以来"],
    "nav": ["单位净值"],
    "acc_nav": ["累计净值"],
}

# 导出/展示时的优先列顺序
PREFERRED_EXPORT_COLS = ["code", "name", "type", "return_1m", "return_3m",
                         "return_6m", "return_1y", "return_3y", "return_ytd"]

NUMERIC_ALIASES = ["return_1w", "return_1m", "return_3m", "return_6m",
                   "return_1y", "return_2y", "return_3y", "return_ytd", "nav", "acc_nav"]


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """根据 COLUMN_MAPPING 解析实际列名，返回 {标准名: 实际列名}。

    先精确匹配，再做受限模糊匹配（要求候选词占列名主体），
    避免 '近1年' 被误匹配到 '近1月' 这类短串误命中。
    """
    col_map: Dict[str, str] = {}
    df_cols = list(df.columns)

    for standard_name, candidates in COLUMN_MAPPING.items():
        for candidate in candidates:
            if candidate in df_cols:
                col_map[standard_name] = candidate
                break
        if standard_name in col_map:
            continue
        for col in df_cols:
            col_str = str(col)
            matched = False
            for candidate in candidates:
                if candidate in col_str and len(col_str) <= len(candidate) + 3:
                    col_map[standard_name] = col
                    matched = True
                    break
            if matched:
                break
    return col_map


# 兼容旧调用
def build_column_map(df: pd.DataFrame) -> Dict[str, str]:
    return resolve_columns(df)


def coerce_numeric(df: pd.DataFrame, col_map: Dict[str, str],
                  aliases: Optional[List[str]] = None) -> pd.DataFrame:
    """把收益类列统一转成数值（无法转换的置为 NaN），原地修改并返回同一 DataFrame。"""
    for alias in aliases or NUMERIC_ALIASES:
        col = col_map.get(alias)
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_float(v: Any, digits: int = 2) -> Optional[float]:
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return None


def pick_column(df: pd.DataFrame, index: int, name_hint: Optional[str] = None) -> Optional[Any]:
    """按位置取列（兼容 akshare 列数变化），必要时用名称提示校验。"""
    if df is None or df.empty or index >= len(df.columns):
        return None
    col = df.columns[index]
    if name_hint and name_hint not in str(col):
        # 名称对不上时再按名称找一次，找不到就用位置结果并记录日志
        match = [c for c in df.columns if name_hint in str(c)]
        if match:
            return match[0]
        logger.debug(f"[列解析] 第 {index} 列名为 {col}，与提示 {name_hint} 不符，按位置使用")
    return col


def normalize_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    把基金代码列强制还原成 6 位零补齐字符串。

    磁盘缓存用 pd.read_json 读回来时，"002910" 会被推断成整数 2910，
    前端显示和"点详情"请求都会带着错的代码（6 位校验直接 400）。
    所有出仓路径都要过这里。缺失代码保持为空，不伪造出 "000000"。
    """
    col_map = resolve_columns(df)
    col = col_map.get("code")
    if not col or col not in df.columns:
        return df

    def fix(value: Any) -> str:
        if value is None or (isinstance(value, float) and value != value):   # NaN
            return ""
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if text.lower() in {"", "nan", "none", "<na>"}:
            return ""
        return text.zfill(6)

    df[col] = [fix(v) for v in df[col]]
    return df


def to_records(df: pd.DataFrame, col_map: Dict[str, str]) -> List[dict]:
    """DataFrame -> 记录列表，标准字段名覆盖原始中文列名。"""
    records = []
    reverse = set(col_map.values())
    for row in df.to_dict(orient="records"):
        item: Dict[str, Any] = {}
        for alias, col in col_map.items():
            val = row.get(col)
            item[alias] = val if not _is_na(val) else None
        for col, val in row.items():
            if col not in reverse:
                item[col] = val if not _is_na(val) else None
        records.append(item)
    return records


def _is_na(val: Any) -> bool:
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False
