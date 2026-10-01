"""提示词。

单独成文件的原因：改提示词是最高频的改动之一，不应该让人在 LLM 调用逻辑里翻。
"""
from __future__ import annotations

SYSTEM_TEMPLATE = """你是一个专业的基金筛选助手。你的能力边界很重要，请严格遵守：

1. 只依据下面提供的数据上下文回答，数据里没有的不要编造（不要臆造代码、收益率）。
2. 涉及具体推荐时，必须同时给出费用与风险的提示，并说明历史收益不代表未来表现。
3. 不做多空/杠杆类建议：公募基金无法做空，杠杆只放大方差而不改变期望。
4. 如果用户的诉求其实需要更长历史或样本外验证才能回答，明确说"需要样本外检验"，
   并建议使用"稳健性体检"功能，而不是直接给结论。

当前数据上下文：
{context}

请用简洁、专业的中文回答。"""

RESPONSE_GUARDRAIL = "\n\n以上内容由模型基于公开数据生成，仅供参考，不构成投资建议。"


def build_context(stats: dict) -> str:
    """把数据概况拼成给模型的上下文（限制长度，避免 token 浪费）。"""
    lines = []
    for key, label in [
        ("as_of", "当前时间"),
        ("fund_count", "基金库数量"),
        ("avg_return_1y", "近1年平均收益(%)"),
        ("max_return_1y", "近1年最高收益(%)"),
        ("benchmark", "基准"),
    ]:
        if stats.get(key) is not None:
            lines.append(f"- {label}：{stats[key]}")
    return "\n".join(lines) if lines else "- 暂无可用数据概况"


def system_prompt(context: str) -> str:
    return SYSTEM_TEMPLATE.format(context=context)
