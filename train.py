"""自训练命令行入口。

用法示例：
    python train.py --codes 002910,110022,161725 --rounds 2
    python train.py --codes 002910,110022 --auto            # 让所有模拟盘账户按冠军规则走一天
    python train.py --review pa-xxxx                        # 打印复盘报告
    python train.py --promote st-xxxx                       # 人工晋升（记录冻结日）

刻意没有 --auto-promote：晋升必须显式做，因为晋升那一刻才是"冻结日"，
也就是样本外计时的起点。让它自动发生，等于把这条纪律删掉。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from agent import training
from analysis.signals import StrategyGenes
from config import load_settings
from container import build_state
from services import paper as paper_svc


def _codes(raw: str):
    return [c.strip() for c in (raw or "").replace(" ", ",").split(",") if c.strip()]


def _print_round(result: dict) -> None:
    gates = result["gates"]
    hold = result["holdout"]
    print(f"\n{'=' * 68}")
    print(f"轮次完成：训练段 {result['split']['train_days']} 天 / "
          f"留出段 {result['split']['holdout_days']} 天 "
          f"({result['split']['holdout_range']})，评估 {result['evolution']['evaluations']} 组参数")
    print(f"  挑战者 {result['record']['id']}  得分 {result['record']['score']}  状态 {result['record']['status']}")
    print(f"  闸门  参数平原={'过' if gates['plateau_ok'] else '不过'}  "
          f"DSR={'过' if gates['dsr_ok'] else '不过'}  "
          f"walk-forward={'过' if gates['walk_forward_ok'] else '不过'}")
    for line in gates["failures"]:
        print(f"    · {line}")
    if hold.get("available"):
        print(f"  留出段首考：收益 {hold['total_return_pct']}%  "
              f"回撤 {hold['max_drawdown_pct']}%  夏普 {hold['sharpe']}  "
              f"超额(vs 定投) {hold['alpha_vs_dca_pct']}%  费用 {hold['cost_pct']}%")
    print(f"  结论：{result['promotion_note']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="基金模拟盘自训练闭环")
    parser.add_argument("--codes", default="", help="参与进化/跟踪的基金代码，逗号分隔")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--pop", type=int, default=14)
    parser.add_argument("--gen", type=int, default=3)
    parser.add_argument("--holdout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--auto", action="store_true", help="让现有账户按冠军规则走一天")
    parser.add_argument("--review", default="", help="打印指定账户的复盘报告")
    parser.add_argument("--promote", default="", help="晋升指定策略为冠军（记录冻结日）")
    parser.add_argument("--open", action="store_true", help="用 --codes 开一个绑定了冠军的新账户")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是可读文本")
    args = parser.parse_args(argv)

    settings = load_settings()
    state = build_state(settings)

    if args.promote:
        result = state.archive.promote(args.promote, note="CLI 人工晋升")
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0 if result.get("ok") else 1

    if args.review:
        account = state.paper.get(args.review)
        if account is None:
            print(f"找不到账户 {args.review}")
            return 1
        report = training.build_review(account, state.archive, state.paper)
        facts = json.dumps(report["facts"], ensure_ascii=False, indent=1)
        print(facts)
        print("\n样本外说明：", report.get("note") or report["facts"].get("结论"))
        return 0

    if args.auto:
        accounts = state.paper.list_accounts()
        if not accounts:
            print("还没有模拟账户，先用 --open 开一个")
            return 1
        genes = None
        champ = state.archive.champion()
        if champ:
            genes = StrategyGenes.from_dict(champ.genes)
            print(f"使用冠军 {champ.id}（冻结于 {champ.frozen_at}）")
        else:
            print("档案里没有冠军，使用默认基因")
        for account in accounts:
            result = paper_svc.auto_run(state.paper, state.repo, settings, account.id,
                                        genes=genes, archive=state.archive)
            stats = result["equity"]
            print(f"  {account.id} {account.name}: 权益 {stats['equity']}  "
                  f"现金 {stats['cash']}  待确认 {stats['pending_orders']}  "
                  f"已结算 {len(result['settled'])} 笔  新下单 {len(result['submitted'])} 笔")
        return 0

    codes = _codes(args.codes)
    if len(codes) < 2:
        parser.error("--codes 至少需要 2 只基金（6 位代码，逗号分隔）")

    if args.open:
        opened = paper_svc.open_account(state.paper, state.repo, settings,
                                        f"自训练盘 {date.today()}", args.capital, codes,
                                        seed_buy=True, archive=state.archive)
        print(f"已开户 {opened['account']['id']}  冻结日 {opened['frozen_at']}")
        print("  ", opened["frozen_meaning"])

    for i in range(max(1, args.rounds)):
        result = training.run_round(state.repo, settings, codes, days=args.days,
                                    pop_size=args.pop, generations=args.gen,
                                    seed=args.seed + i, holdout_days=args.holdout,
                                    archive=state.archive)
        if args.json:
            print(json.dumps(result, ensure_ascii=False)[:4000])
        else:
            _print_round(result)

    summary = state.archive.summary()
    print(f"\n档案：共 {summary['total']} 条，冠军 {summary['champions']}，"
          f"挑战者 {summary['challengers']}，被拒 {summary['rejected']}，退役 {summary['retired']}")
    if summary["champion"]:
        print(f"当前冠军：{summary['champion']['id']}  冻结于 {summary['champion']['frozen_at']}")
    print("下一步：python train.py --auto 让账户按冠军规则走一天；"
          "python train.py --review <账户id> 看复盘。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
