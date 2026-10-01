"""自我训练闭环：进化 → 闸门 → 入库 → 留出段验证 → 模拟盘跟踪 → 复盘。

分工是硬性的：**所有数字都由代码算，LLM 只负责把结论写成人话**。
LLM 不参与打分、不参与筛选、不决定晋升 —— 它连"这只基金好不好"都不判断。

一条不可绕过的设计：进化时把最近 holdout_days 个交易日**整段藏起来**，
挑战者只准用更早的数据进化，然后在藏起来的那段上第一次公开成绩。
这才是"自己训练自己"有意义的部分；否则训练一百轮，得到的也只是
同一段历史的第一百种过拟合方式。
"""
from __future__ import annotations

from datetime import date as _date
from typing import Dict, List, Optional

import pandas as pd

from analysis import validation as VAL
from analysis.archive import CHALLENGER, Gates, StrategyArchive, StrategyRecord
from analysis.risk_rules import DEFAULT_EXIT, ExitConfig
from analysis.search import evolve, fitness, plateau_check
from analysis.simulation import simulate
from analysis.signals import DEFAULT_ENV, Environment, StrategyGenes
from config import Settings
from core.errors import InsufficientData, InvalidInput
from paper.account import PaperAccount
from paper.engine import step as paper_step
from paper.store import PaperStore
from sources.repository import FundRepository

TRAIN_DAYS_MIN = 250
HOLDOUT_DAYS_DEFAULT = 60


def _split(returns_df: pd.DataFrame, holdout_days: int):
    holdout = max(int(holdout_days), 20)
    if len(returns_df) < TRAIN_DAYS_MIN + holdout:
        raise InsufficientData(
            f"样本 {len(returns_df)} 天不足以切出训练段与留出段"
            f"（需要至少 {TRAIN_DAYS_MIN + holdout} 天）")
    return returns_df.iloc[:-holdout], returns_df.iloc[-holdout:]


def build_gates(plateau: Dict[str, object], dsr: Dict[str, object],
                walk_forward: Dict[str, object]) -> Gates:
    return Gates(
        plateau_ok=bool(plateau.get("available") and plateau.get("on_plateau")),
        plateau_detail=plateau,
        dsr_ok=bool(dsr.get("available") and dsr.get("robust_at_95")),
        dsr_detail=dsr,
        walk_forward_ok=bool(walk_forward.get("available") and walk_forward.get("effective_out_of_sample")),
        walk_forward_detail=walk_forward,
    )


def run_round(repo: FundRepository, settings: Settings, codes: List[str],
              days: int = 1200, pop_size: int = 14, generations: int = 3,
              seed: int = 2026, holdout_days: int = HOLDOUT_DAYS_DEFAULT,
              env: Environment = DEFAULT_ENV,
              exit_cfg: Optional[ExitConfig] = None,
              archive: Optional[StrategyArchive] = None) -> Dict[str, object]:
    """
    跑一轮自训练。返回挑战者、三重闸门结果、留出段成绩、以及能否晋升的建议。
    """
    from services.portfolio import clean_codes
    codes = clean_codes(codes, min_count=2)
    returns_df, nav_by_code = repo.require_aligned(codes, days)
    exit_cfg = exit_cfg or DEFAULT_EXIT
    archive = archive or StrategyArchive()

    train, holdout = _split(returns_df, holdout_days)
    holdout_dates = [d.strftime("%Y-%m-%d") for d in holdout.index]

    report = evolve(train, env=env, exit_cfg=exit_cfg, pop_size=pop_size,
                    generations=generations, seed=seed)
    challenger_genes = StrategyGenes.from_dict(report["challenger"]["genes"])

    plateau = plateau_check(challenger_genes, train, env=env, exit_cfg=exit_cfg)
    wf = VAL.walk_forward(nav_by_code, train_days=min(252, len(train) // 2),
                          test_days=max(21, holdout_days), folds=4)
    final_review = report.get("final_review", {}) or {}
    dsr = final_review.get("deflated_sharpe", {}) or {}
    gates = build_gates(plateau, dsr, wf)

    # 留出段：进化过程从未见过这段，所以这里是这个策略的第一次"未见数据"考试
    holdout_result = simulate(holdout, genes=challenger_genes, env=env, exit_cfg=exit_cfg)
    holdout_summary = _holdout_summary(holdout_result)
    baseline = simulate(holdout, genes=StrategyGenes(), env=env, exit_cfg=exit_cfg)

    record = StrategyRecord(
        id=StrategyRecord.new_id(), created_at=_date.today().isoformat(),
        genes=challenger_genes.to_dict(), score=float(report["challenger"]["score"]),
        breakdown=report["challenger"].get("breakdown", {}), gates=gates,
        n_trials=int(report.get("n_trials", 0)),
        window=[holdout_dates[0], holdout_dates[-1]] if holdout_dates else [],
        notes=f"训练段 {len(train)} 天 / 留出段 {len(holdout)} 天",
        review={"holdout": holdout_summary},
    )
    stored = archive.add(record)

    return {
        "record": stored.to_dict(),
        "gates": gates.to_dict(),
        "promotable": gates.passed,
        "promotion_note": (
            "闸门通过，可人工晋升；晋升当天记为冻结日，此后净值才算样本外证据"
            if gates.passed else
            "闸门未通过，已按 rejected 入库；分数高不代表可信：" + "；".join(gates.failures())),
        "holdout": holdout_summary,
        "holdout_baseline_equalweight": _holdout_summary(baseline),
        "evolution": {"evaluations": report.get("evaluations"),
                      "history": report.get("history"),
                      "top": report.get("top", [])[:3],
                      "final_review": final_review},
        "split": {"train_days": len(train), "holdout_days": len(holdout),
                  "holdout_range": [holdout_dates[0], holdout_dates[-1]] if holdout_dates else None},
        "codes": codes,
    }


def _holdout_summary(result: Dict[str, object]) -> Dict[str, object]:
    if "error" in result:
        return {"available": False, "note": result["error"]}
    res = result["result"]
    st = res["strategy"]
    m = st["metrics"]
    return {
        "available": True,
        "total_return_pct": m.get("total_return_pct"),
        "max_drawdown_pct": m.get("max_drawdown_pct"),
        "sharpe": m.get("sharpe"),
        "alpha_vs_dca_pct": res.get("alpha_vs_dca_pct"),
        "avg_exposure_pct": res.get("avg_exposure_pct"),
        "trades": st.get("trade_count"),
        "cost_pct": st.get("total_cost_pct"),
        "fused": res.get("fused"),
    }


def advance_paper_account(account: PaperAccount, repo: FundRepository,
                          record: StrategyArchive | None = None,
                          genes: Optional[StrategyGenes] = None,
                          on_date: Optional[str] = None) -> Dict[str, object]:
    """让绑定了策略的模拟盘走一天：先结算待确认订单，再按规则下单。"""
    codes = sorted({l.code for l in account.lots} | {o.code for o in account.orders})
    if not codes:
        raise InvalidInput("账户还没有任何标的，请先申购或指定跟踪的基金代码")
    nav_by_code = {c: repo.nav_history(c, days=1200) for c in codes}
    nav_by_code = {c: s for c, s in nav_by_code.items() if len(s) > 1}
    if not nav_by_code:
        raise InsufficientData("所有标的的净值都取不到，无法结算")
    latest = max(s.index.max() for s in nav_by_code.values())
    day = on_date or latest.strftime("%Y-%m-%d")
    result = paper_step(account, day, nav_by_code, genes or StrategyGenes())
    return result


def build_review(account: PaperAccount, archive: StrategyArchive,
                 store: PaperStore) -> Dict[str, object]:
    """
    复盘报告：数字全部由代码算好，LLM 只拿到这些数字去写叙述。

    最关键的一栏是 out_of_sample —— 它明确告诉你这份成绩单里
    有多少天是策略"没见过"的。0 天就直接写明"不构成验证"。
    """
    stats = account.stats()
    champ = archive.champion()
    oos = stats.get("out_of_sample", {})
    positions = account.positions(_date.today().isoformat(), {})
    facts = {
        "账户": account.name,
        "初始资金": stats["initial_capital"],
        "当前权益": stats["equity"],
        "现金": stats["cash"],
        "持仓市值": stats["positions_value"],
        "累计收益_pct": (stats["metrics"].get("total_return_pct")
                         if "error" not in stats["metrics"] else None),
        "最大回撤_pct": (stats["metrics"].get("max_drawdown_pct")
                         if "error" not in stats["metrics"] else None),
        "夏普": (stats["metrics"].get("sharpe") if "error" not in stats["metrics"] else None),
        "已确认交易数": stats["confirmed_trades"],
        "累计已付费用": stats["fees_paid"],
        "费用占初始资金_pct": stats["fee_drag_pct"],
        "样本外天数": oos.get("days_observed", 0),
        "样本外收益_pct": oos.get("return_pct"),
        "策略冻结日": oos.get("frozen_at"),
        "冠军策略": ({"id": champ.id, "score": champ.score, "frozen_at": champ.frozen_at,
                     "gates_passed": champ.gates.passed} if champ else None),
    }
    facts["结论"] = (
        f"样本外仅观察 {facts['样本外天数']} 天"
        if not oos.get("evidence") else
        f"样本外已观察 {facts['样本外天数']} 天，收益 {facts['样本外收益_pct']}%"
    )
    return {"facts": facts, "account_id": account.id,
            "evidence": bool(oos.get("evidence")),
            "note": oos.get("note", ""),
            "positions": positions,
            "generated_at": _date.today().isoformat()}


REVIEW_PROMPT = """你是量化复盘助手。下面是一段**已经由代码算好的**模拟盘事实，
请只基于这些事实写一段不超过 200 字的中文复盘，不要新增任何数字，
不要给投资建议，不要美化样本外天数不足的事实。若样本外天数为 0，
必须明确说明"当前成绩全部来自样本内，不构成验证"。

事实：
{facts}
"""


async def narrate_review(review: Dict[str, object], settings: Settings, llm=None) -> Dict[str, object]:
    """可选：让 LLM 把事实写成叙述。失败时退回模板，绝不影响数字部分。"""
    import json
    facts = json.dumps(review["facts"], ensure_ascii=False, default=str)
    template = (f"{review['facts'].get('账户')}：当前权益 {review['facts'].get('当前权益')}，"
                f"累计收益 {review['facts'].get('累计收益_pct')}%，"
                f"已付费用 {review['facts'].get('累计已付费用')}。"
                f"{review['note'] or review['facts'].get('结论')}")
    if llm is None or not getattr(llm, "enabled", False):
        return {"text": template, "engine": "template"}
    try:
        text = await llm.chat([
            {"role": "system", "content": REVIEW_PROMPT.format(facts=facts)},
            {"role": "user", "content": "请输出复盘正文。"},
        ])
        return {"text": text, "engine": "llm"}
    except Exception as e:
        return {"text": template, "engine": "template", "llm_error": str(e)[:120]}
