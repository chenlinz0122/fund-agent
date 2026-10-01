"""规则引擎处理函数。

每个意图一个函数，返回 None 表示"这个我答不了"，由 ChatService 继续降级。
新增能力 = 写一个 handler + 在 intent.INTENTS 里加一行关键词，不用读懂别的分支。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from agent import intent as I
from agent.knowledge import KnowledgeBase, DISCLAIMER
from analysis import expectancy as exp
from analysis import metrics as M
from analysis import metrics as M
from config import Settings
from services import funds as funds_service
from services import screening as scr
from sources.repository import FundRepository


@dataclass
class AgentContext:
    """规则引擎所需的全部依赖，显式注入（不再有模块级全局）。"""
    settings: Settings
    repo: FundRepository
    kb: KnowledgeBase


# ========== 文本格式化 ==========
def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def fund_line(i: int, f: dict, show_type: bool = False) -> str:
    code = f.get("code", "N/A")
    name = f.get("name", "N/A")
    ret = f.get("return_1y", "N/A")
    extra = ""
    if show_type and f.get("type"):
        extra = f" [{f['type']}]"
    return f"{i}. {name}（{code}）{extra} 近1年收益：{ret}%\n"


def fetch_fund_detail(code: str, ctx: AgentContext) -> dict:
    """单点封装，便于测试替换与统一异常处理。"""
    try:
        return funds_service.fund_detail(code, ctx.repo, ctx.settings)
    except Exception as e:  # 详情抓取失败不应拖垮整条回答
        return {"code": code, "info": {}, "metrics": {}, "error": str(e)}


# ========== 各意图处理函数 ==========
def handle_market(message: str, df: Optional[pd.DataFrame], ctx: AgentContext) -> Optional[str]:
    payload = ctx.repo.news_payload()
    news, boards = payload.get("news") or [], payload.get("boards") or []
    if not news and not boards:
        return "暂时无法获取实时市场资讯，可能是网络受限。请稍后重试，或在仪表盘查看基金数据。"
    lines: List[str] = []
    if boards:
        lines.append("📊 板块异动（涨跌幅）：")
        for b in boards[:10]:
            chg = b.get("change_pct")
            chg_str = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "N/A"
            lines.append(f"  • {b.get('name', '')}  {chg_str}")
        lines.append("")
    if news:
        lines.append("📰 最新财经快讯：")
        for i, n in enumerate(news[:12], 1):
            tag = f"[{n['tag']}] " if n.get("tag") else ""
            lines.append(f"{i}. {tag}{(n.get('summary') or '')[:80]}")
        lines.append("\n提示：以上资讯来自公开财经接口，仅供参考，不构成投资建议。")
    return "\n".join(lines)


def handle_concept(message: str, df: Optional[pd.DataFrame], ctx: AgentContext) -> Optional[str]:
    return ctx.kb.answer(message)


def handle_fund_type_overview(message: str, df: Optional[pd.DataFrame],
                              ctx: AgentContext) -> Optional[str]:
    hits = ctx.kb.search("基金的分类", top_k=1)
    if hits:
        return hits[0].get("content", "") + "\n\n（来源：内置基金知识库，仅供参考。）"
    return (
        "常见基金类型：\n"
        "• 股票型基金：≥80%投资股票，高风险高收益\n"
        "• 债券型基金：≥80%投资债券，中低风险\n"
        "• 混合型基金：股债灵活配置，中等风险\n"
        "• 指数型基金：跟踪特定指数，被动管理\n"
        "• 货币基金：投资短期货币工具，低风险\n"
        "• QDII基金：投资海外市场\n"
    )


def handle_recommend(message: str, df: Optional[pd.DataFrame], ctx: AgentContext) -> Optional[str]:
    if df is None:
        return None
    ftype = I.extract_fund_type_from_message(message)
    threshold = I.extract_return_threshold(message)
    criteria = scr.FundCriteria(
        fund_type=ftype,
        min_return_1y=threshold if threshold is not None else 10,
        sort_by="return_1y", sort_order="desc", limit=10,
    )
    res = scr.apply_criteria(df, criteria)
    if not res.funds:
        return None
    scope = f"{ftype}基金" if ftype else "基金"
    out = f"根据近 1 年收益表现，以下{scope}排名靠前：\n\n"
    out += "".join(fund_line(i, f) for i, f in enumerate(res.funds[:5], 1))
    out += ("\n⚠️ 注意：这就是「只看涨幅排名」，本质是追涨。历史收益不代表未来表现，"
            "建议结合最大回撤、波动率与费后期望再判断。")
    return out


def handle_compare(message: str, df: Optional[pd.DataFrame], ctx: AgentContext) -> Optional[str]:
    codes = I.extract_codes(message, limit=6)
    if len(codes) < 2:
        return None
    out = f"正在对比 {len(codes[:3])} 只基金：\n\n"
    for code in codes[:3]:
        detail = fetch_fund_detail(code, ctx)
        name = (detail.get("info") or {}).get("基金简称", code)
        m = detail.get("metrics") or {}
        e = detail.get("expectancy") or {}
        out += f"**{name}（{code}）**\n"
        out += f"  近1年收益：{m.get('total_return_1y', 'N/A')}%\n"
        out += f"  年化波动率：{m.get('volatility', 'N/A')}%\n"
        out += f"  夏普比率：{m.get('sharpe_ratio', 'N/A')}\n"
        out += f"  最大回撤：{m.get('max_drawdown', 'N/A')}%\n"
        if e:
            out += (f"  费后年化：{e.get('net_annual_return_pct', 'N/A')}%"
                    f"（费用侵蚀 {e.get('annualized_fee_drag_pct', 'N/A')}%/年）\n")
        out += "\n"
    return out


def handle_risk(message: str, df: Optional[pd.DataFrame], ctx: AgentContext) -> Optional[str]:
    if df is None:
        return None
    criteria = scr.FundCriteria(
        fund_type=I.extract_fund_type_from_message(message),
        min_return_1y=5, max_return_1y=30,
        sort_by="return_1y", sort_order="desc", limit=20,
    )
    res = scr.apply_criteria(df, criteria)
    if not res.funds:
        return None
    out = "追求稳健收益，以下基金收益适中、波动相对可控：\n\n"
    out += "".join(fund_line(i, f) for i, f in enumerate(res.funds[:5], 1))
    out += "\n提示：建议关注最大回撤和波动率指标，选择与自身风险承受能力匹配的产品。"
    return out


def handle_expectancy(message: str, df: Optional[pd.DataFrame], ctx: AgentContext) -> Optional[str]:
    """费后期望：回答"买这个到底划不划算"。"""
    cfg = ctx.settings.fees
    trips = I.extract_round_trips(message) or cfg.default_round_trips_per_year
    ftype = I.extract_fund_type_from_message(message)
    profile = exp.fee_profile(f"{ftype}型" if ftype else None, cfg)
    drag = exp.fee_drag(profile, holding_years=cfg.default_holding_years,
                        round_trips_per_year=trips,
                        penalty_days=cfg.penalty_days, penalty_redemption=cfg.penalty_redemption)

    codes = I.extract_codes(message, limit=1)
    head = ""
    if codes:
        detail = fetch_fund_detail(codes[0], ctx)
        m = detail.get("metrics") or {}
        # 用用户问到的换手次数重新计算，而不是沿用详情里按默认换手算好的结果，
        # 否则同一段回答里"结论"和"费用结构"会互相矛盾。
        history = detail.get("history") or []
        returns = None
        if len(history) > 2:
            nav = pd.Series({str(h["date"]): h["nav"] for h in history}, dtype=float).sort_index()
            returns = M.to_returns(nav)
        raw = {
            "annual_return_pct": m.get("annual_return"),
            "volatility_pct": m.get("volatility"),
            "sharpe": m.get("sharpe_ratio"),
            "data_points": m.get("data_points"),
        }
        report = exp.expectancy_report(
            raw, fund_type=(detail.get("info") or {}).get("基金类型") or ftype,
            returns=returns, holding_years=cfg.default_holding_years,
            round_trips_per_year=trips, fee_cfg=cfg, analysis_cfg=ctx.settings.analysis,
        )
        head = (f"**{codes[0]}** 毛年化 {report['gross_annual_return_pct']}%，"
                f"费后年化 {report['net_annual_return_pct']}%，净夏普 {report['net_sharpe']}。\n"
                f"结论：{report['verdict']}\n\n")
        if report["risk_of_ruin_notes"]:
            head += f"命中的风险点：{'、'.join(report['risk_of_ruin_notes'])}\n\n"

    out = head + (
        f"费用结构（{profile.fund_type}，假设每年往返 {trips:g} 次）：\n"
        f"• 申购费 {profile.subscription}% + 赎回费 {profile.redemption}%\n"
        f"• 管理+托管 {profile.annual_internal:g}%/年（每年从净值计提，无法回避）\n"
        f"• 折算到每年的总拖累约 {drag['annualized_drag_pct']}%\n"
    )
    if drag.get("penalty_applies"):
        out += f"• ⚠️ 平均持有不足 {cfg.penalty_days} 天，触发惩罚性赎回费 {cfg.penalty_redemption}%\n"
    out += (
        "\n为什么这很重要：交易成本和管理费是**确定性负期望**，"
        "而超额收益是不确定的。一个毛期望略正的策略，"
        "在足够高的换手下净期望必然为负。\n"
        "另外：做空在期望上为负（借券成本 + 向上风险溢价 + 亏损上不封顶），"
        "杠杆只放大方差不改变期望 —— 公募基金天然 long-only，请不要用任何形式加杠杆。"
    )
    return out + DISCLAIMER


def handle_ranking(message: str, df: Optional[pd.DataFrame], ctx: AgentContext) -> Optional[str]:
    if df is None:
        return None
    ftype = I.extract_fund_type_from_message(message)
    res = scr.apply_criteria(df, scr.FundCriteria(
        fund_type=ftype, sort_by="return_1y", sort_order="desc", limit=10))
    if not res.funds:
        return None
    scope = f"{ftype}基金" if ftype else "全部基金"
    out = f"{scope}近 1 年收益排行榜 Top {min(10, len(res.funds))}：\n\n"
    out += "".join(fund_line(i, f, show_type=True) for i, f in enumerate(res.funds, 1))
    out += ("\n⚠️ 榜单只描述过去，不含任何费用与风险调整；"
            "这类单一收益排名的样本外预测力需要用「稳健性体检」验证。")
    return out


def handle_strategy(message: str, df: Optional[pd.DataFrame], ctx: AgentContext) -> Optional[str]:
    """仓位策略与止盈止损：解释"降次"的思路，并给出当前各天平的实际读数。"""
    from analysis.risk_rules import FUND_EXIT_REALITY, describe_rules
    from services import simulation as sim_service

    genes = sim_service.build_genes(None)
    cfg = sim_service.build_exit(None)
    rules = describe_rules(genes, ctx.settings and _default_env(), cfg)
    p = rules["position_policy"]

    out = [
        "这个系统的仓位不是预测出来的，是把风险状态映射成敞口档位（降次求解）：",
        f"• 波动天平：目标年化 {p['vol_target_pct']}%，超出容差 {p['vol_tolerance_pct']}% 才开始减仓",
        f"• 回撤天平：{p['drawdown_bands_pct'][0]}% / {p['drawdown_bands_pct'][1]}% 两档递减",
        f"• 加速度天平：ABG 滤波器给出的二阶导（下跌在加速则减仓）",
        f"• 摩擦控制：仓位偏差不到 {p['no_trade_band_pct']}% 不调仓；每 {p['rebalance_stride']} 个交易日复核一次",
        f"• 硬红线：杠杆上限 {p['leverage_cap']}（不加杠杆）、long_only={p['long_only']}、"
        f"全局回撤熔断 {rules['exit_rules'][1]['detail']}",
        "",
        "关于止盈止损，基金和期货有三点本质差别（这是最容易翻车的地方）：",
    ]
    for item in FUND_EXIT_REALITY[:4]:
        out.append(f"  - {item['item']}：{item['effect']}")

    codes = I.extract_codes(message, limit=3)
    if codes:
        out.append("")
        out.append("当前状态读数：")
        try:
            snap = sim_service.policy_snapshot(codes, ctx.repo, days=365)
            for row in snap["rows"]:
                if not row.get("available"):
                    out.append(f"  • {row['code']}：{row.get('note', '数据不足')}")
                    continue
                exposure = row["target_exposure"]["exposure"] * 100
                out.append(f"  • {row['code']} 建议敞口 {exposure:.0f}%"
                           f"（{'；'.join(row['target_exposure']['reasons'][:2])}）")
        except Exception as e:
            out.append(f"  • 状态计算失败：{e}")

    out.append("")
    out.append("另外提醒：参数是可以用遗传算法搜出来的，但搜得越多，最好的那个分数越可能是噪声。"
               "所以进化结果默认不自动晋升，且必须看 Deflated Sharpe 与参数平原，而不是看最高分。")
    return "\n".join(out) + DISCLAIMER


def _default_env():
    from analysis.signals import DEFAULT_ENV
    return DEFAULT_ENV


HANDLERS: Dict[str, Callable] = {
    "market": handle_market,
    "concept": handle_concept,
    "fund_type_overview": handle_fund_type_overview,
    "recommend": handle_recommend,
    "compare": handle_compare,
    "strategy": handle_strategy,
    "risk": handle_risk,
    "expectancy": handle_expectancy,
    "ranking": handle_ranking,
}


def default_help() -> str:
    return """我可以帮你：
• 推荐基金：输入"推荐基金"或"买什么基金好"
• 对比基金：输入"对比 000001 000002"（替换为实际基金代码）
• 稳健投资：输入"稳健型基金"或"低风险基金"
• 收益排行：输入"收益排行榜"
• 费后期望：输入"000001 划不划算""持有 3 个月赎回费多少"
• 仓位与止盈止损：输入"该怎么控制仓位""基金止损有用吗""000001 现在该持多少仓"
• 概念科普：输入"什么是夏普比率""基金定投""如何选基金"等
• 市场资讯：输入"今日热点""看看市场新闻""哪些板块活跃"

也可以在仪表盘设置筛选条件精确筛选，或在组合分析页做回测、稳健性体检与策略模拟。"""
