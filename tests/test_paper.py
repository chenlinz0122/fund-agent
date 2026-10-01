"""模拟盘账本、策略档案与自动交易的测试（全部离线，用合成净值）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd
import pytest

from analysis.archive import CHALLENGER, REJECTED, Gates, StrategyArchive, StrategyRecord
from config import FeeConfig
from paper.account import BUY, CONFIRMED, PENDING, REJECTED as ORDER_REJECTED, SELL, PaperAccount
from paper.engine import decide, portfolio_nav, step
from paper.store import PaperStore
from analysis.signals import StrategyGenes


@pytest.fixture
def fees():
    return FeeConfig()


def _acct(fees, capital=100000.0, frozen_at=None):
    return PaperAccount.create("测试盘", capital, on_date="2026-01-05", fees=fees,
                              frozen_at=frozen_at or "2025-12-01")


class TestTPlusOneSettlement:
    def test_order_not_filled_same_day(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 10000, "2026-01-05")
        confirmed = acct.settle("2026-01-05", {"002910": 1.5})
        assert confirmed == []
        assert acct.orders[0].status == PENDING
        assert acct.shares_of("002910") == 0

    def test_confirms_next_day_at_published_nav(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 10000, "2026-01-05")
        done = acct.settle("2026-01-06", {"002910": 1.5})
        assert len(done) == 1
        order = done[0]
        assert order.status == CONFIRMED and order.nav == 1.5
        # 混合型申购费 0.15%
        assert abs(order.filled_shares - (10000 * (1 - 0.0015) / 1.5)) < 0.01
        assert abs(acct.cash - 90000) < 1e-6

    def test_cash_locked_on_submission(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 30000, "2026-01-05")
        assert abs(acct.cash - 70000) < 1e-6

    def test_insufficient_cash_rejected(self, fees):
        acct = _acct(fees)
        order = acct.submit_buy("002910", 999999, "2026-01-05")
        assert order.status == ORDER_REJECTED
        assert "现金" in order.note

    def test_bad_code_rejected(self, fees):
        acct = _acct(fees)
        assert acct.submit_buy("abc", 100, "2026-01-05").status == ORDER_REJECTED

    def test_waits_when_nav_missing(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 1000, "2026-01-05")
        assert acct.settle("2026-01-06", {}) == []
        assert acct.orders[0].status == PENDING


class TestRedemptionFeeTiers:
    def _buy_one(self, acct, opened="2026-01-06", nav=1.0, amount=10000):
        acct.submit_buy("002910", amount, "2026-01-05")
        acct.settle(opened, {"002910": nav})

    def test_penalty_within_7_days(self, fees):
        acct = _acct(fees)
        self._buy_one(acct)
        acct.submit_sell("002910", None, "2026-01-08")
        done = acct.settle("2026-01-09", {"002910": 1.0})[0]
        assert done.fee == pytest.approx(10000 * (1 - 0.0015) * 0.015, abs=1.0)

    def test_no_penalty_after_a_year(self, fees):
        acct = _acct(fees)
        self._buy_one(acct)
        acct.submit_sell("002910", None, "2027-06-01")
        done = acct.settle("2027-06-02", {"002910": 1.0})[0]
        assert done.fee == pytest.approx(0.0, abs=0.01)

    def test_oldest_lot_consumed_first(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 5000, "2026-01-05")
        acct.settle("2026-01-06", {"002910": 1.0})
        acct.submit_buy("002910", 5000, "2026-05-05")
        acct.settle("2026-05-06", {"002910": 1.0})
        assert len(acct.lots) == 2
        acct.submit_sell("002910", 5000, "2026-05-08")
        acct.settle("2026-05-11", {"002910": 1.0})
        assert acct.lots[0].opened_date == "2026-05-06"      # 一月的老批次先被赎回，只剩五月批次
        # 老批次 2026-01-06 持有到 2026-05-11 = 125 天 -> 落在 (30,365) 档 = 0.5%
        assert acct.orders[-1].fee == pytest.approx(5000 * 0.005, rel=0.02)

    def test_positions_report_holding_days_and_rate(self, fees):
        acct = _acct(fees)
        self._buy_one(acct)
        rows = acct.positions("2026-01-20", {"002910": 1.1})
        assert rows[0]["holding_days"] == 14
        assert rows[0]["redemption_rate_pct"] == 0.75
        assert rows[0]["penalty_applies"] is False
        assert rows[0]["pnl_pct"] > 9


class TestEquityAndExposure:
    def test_mark_to_market_tracks_nav(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 50000, "2026-01-05")
        acct.settle("2026-01-06", {"002910": 1.0})
        acct.settle("2026-01-07", {"002910": 1.2})
        last = acct.equity_history[-1]
        assert last["holdings"] == pytest.approx(50000 * (1 - 0.0015) * 1.2, rel=1e-3)
        assert last["exposure_pct"] == pytest.approx(
            last["holdings"] / last["equity"] * 100, abs=0.2)

    def test_stats_include_fees_and_metrics(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 20000, "2026-01-05")
        acct.settle("2026-01-06", {"002910": 1.0})
        stats = acct.stats()
        assert stats["fees_paid"] > 0
        assert stats["confirmed_trades"] == 1
        assert "out_of_sample" in stats


class TestOutOfSampleDiscipline:
    def test_no_frozen_date_means_no_evidence(self, fees):
        acct = PaperAccount.create("无冻结", 1000, "2026-01-05", fees=fees)
        out = acct.out_of_sample_stats()
        assert out["evidence"] is False and out["frozen_at"] is None

    def test_zero_days_after_freeze_is_not_evidence(self, fees):
        acct = _acct(fees, frozen_at="2026-01-05")
        out = acct.out_of_sample_stats()
        assert out["evidence"] is False
        assert "样本内" in out["note"]

    def test_days_after_freeze_counted(self, fees):
        acct = _acct(fees, frozen_at="2026-01-05")
        acct.submit_buy("002910", 10000, "2026-01-06")
        for day, nav in [("2026-01-07", 1.0), ("2026-01-08", 1.05), ("2026-01-09", 1.02)]:
            acct.settle(day, {"002910": nav})
        out = acct.out_of_sample_stats()
        assert out["evidence"] is True
        assert out["days_observed"] == 3
        # 总权益含 9 万未投入的现金：10000 元仓位涨 5% 再回落到 2%，
        # 对整个账户只体现为约 0.2% —— 样本外收益必须按账户整体口径算
        assert out["return_pct"] == pytest.approx((100185 / 99985 - 1) * 100, abs=0.05)
        assert out["confirmed_trades"] == 1


class TestSerialization:
    def test_roundtrip(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 1000, "2026-01-05")
        acct.settle("2026-01-06", {"002910": 1.0})
        clone = PaperAccount.from_dict(json.loads(json.dumps(acct.to_dict())))
        assert clone.id == acct.id and len(clone.lots) == 1 and len(clone.orders) == 1
        assert clone.cash == acct.cash

    def test_store_persists(self, tmp_path, fees):
        store = PaperStore(tmp_path / "pa.json")
        acct = _acct(fees)
        assert store.save(acct)
        assert store.get(acct.id).name == "测试盘"
        assert len(store.list_accounts()) == 1
        store.remove(acct.id)
        assert store.get(acct.id) is None

    def test_corrupt_file_degrades_to_empty(self, tmp_path):
        p = tmp_path / "pa.json"
        p.write_text("{ broken", encoding="utf-8")
        assert PaperStore(p).list_accounts() == []


class TestArchiveGates:
    def _gates(self, ok=True):
        return Gates(plateau_ok=ok, plateau_detail={"on_plateau": ok},
                     dsr_ok=ok, dsr_detail={"robust_at_95": ok},
                     walk_forward_ok=ok, walk_forward_detail={"effective_out_of_sample": ok})

    def test_failed_gates_are_rejected_not_stored_as_challenger(self, tmp_path):
        archive = StrategyArchive(tmp_path / "sa.json")
        rec = StrategyRecord(id=StrategyRecord.new_id(), created_at="2026-01-05",
                             genes={"vol_target": 0.2}, score=9.9, gates=self._gates(False))
        stored = archive.add(rec)
        assert stored.status == REJECTED
        assert len(stored.notes) > 0
        assert archive.champion() is None

    def test_promote_requires_passed_gates(self, tmp_path):
        archive = StrategyArchive(tmp_path / "sa.json")
        rec = archive.add(StrategyRecord(id=StrategyRecord.new_id(), created_at="2026-01-05",
                                         genes={}, score=1, gates=self._gates(False)))
        result = archive.promote(rec.id)
        assert result["ok"] is False and "闸门" in result["reason"]

    def test_promote_sets_frozen_at_and_retires_previous(self, tmp_path):
        archive = StrategyArchive(tmp_path / "sa.json")
        a = archive.add(StrategyRecord(id=StrategyRecord.new_id(), created_at="2026-01-05",
                                       genes={}, score=1, gates=self._gates()))
        b = archive.add(StrategyRecord(id=StrategyRecord.new_id(), created_at="2026-01-06",
                                       genes={}, score=2, gates=self._gates()))
        assert archive.promote(a.id, on_date="2026-02-01")["ok"]
        assert archive.champion().id == a.id
        assert archive.champion().frozen_at == "2026-02-01"
        archive.promote(b.id, on_date="2026-03-01")
        assert archive.champion().id == b.id
        assert archive.get(a.id).status == "retired"
        assert archive.get(a.id).retired_at == "2026-03-01"

    def test_double_promote_blocked(self, tmp_path):
        archive = StrategyArchive(tmp_path / "sa.json")
        rec = archive.add(StrategyRecord(id=StrategyRecord.new_id(), created_at="2026-01-05",
                                         genes={}, score=1, gates=self._gates()))
        assert archive.promote(rec.id)["ok"]
        assert archive.promote(rec.id)["ok"] is False

    def test_summary_counts(self, tmp_path):
        archive = StrategyArchive(tmp_path / "sa.json")
        archive.add(StrategyRecord(id=StrategyRecord.new_id(), created_at="2026-01-05",
                                   genes={}, score=1, gates=self._gates()))
        s = archive.summary()
        assert s["challengers"] == 1 and s["champion"] is None

    def test_genes_roundtrip_through_archive(self, tmp_path):
        """档案存的是 dict，重建必须走 from_dict（abg 是嵌套 dict）。"""
        from analysis.abg import ABGParams
        archive = StrategyArchive(tmp_path / "sa.json")
        genes = StrategyGenes(vol_target=0.22, abg=ABGParams(alpha=0.4, k_stop=6.0))
        archive.add(StrategyRecord(id=StrategyRecord.new_id(), created_at="2026-01-05",
                                   genes=genes.to_dict(), score=1, gates=self._gates()))
        stored = archive.records()[0]
        rebuilt = StrategyGenes.from_dict(stored.genes)
        assert rebuilt.vol_target == 0.22
        assert isinstance(rebuilt.abg, ABGParams)
        assert rebuilt.abg.k_stop == 6.0
        assert rebuilt.clipped().vol_target == 0.22

    def test_from_dict_ignores_unknown_and_environment_fields(self):
        rebuilt = StrategyGenes.from_dict({"vol_target": 0.2, "max_leverage": 9.0,
                                           "long_only": False, "junk": 1,
                                           "abg": {"alpha": 0.5, "nope": 2}})
        assert rebuilt.vol_target == 0.2
        assert not hasattr(rebuilt, "max_leverage")
        assert rebuilt.abg.alpha == 0.5


class TestEngineDecisions:
    def _navs(self, periods=400, drift=0.0004, vol=0.012, seed=4):
        dates = pd.bdate_range("2024-01-02", periods=periods)
        rng = np.random.default_rng(seed)
        return {"002910": pd.Series(np.cumprod(1 + rng.normal(drift, vol, periods)), index=dates)}

    def test_short_history_makes_no_orders(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 1000, "2026-01-05")
        acct.settle("2026-01-06", {"002910": 1.0})
        plan = decide(acct, "2026-01-07", self._navs(periods=10))
        assert plan["orders"] == [] and "不足" in plan["reasons"][0]

    def test_no_trade_band_suppresses_tiny_delta(self, fees):
        acct = _acct(fees)
        navs = self._navs()
        acct.submit_buy("002910", 90000, "2026-01-05")
        acct.settle("2026-01-06", {"002910": float(navs["002910"].iloc[-1])})
        genes = StrategyGenes(min_trade_delta=0.99, vol_target=0.99)
        assert decide(acct, "2026-01-07", navs, genes)["orders"] == []

    def test_high_vol_reduces_target_exposure(self, fees):
        acct = _acct(fees)
        navs = self._navs(vol=0.03)
        acct.submit_buy("002910", 90000, "2026-01-05")
        acct.settle("2026-01-06", {"002910": float(navs["002910"].iloc[-1])})
        plan = decide(acct, "2026-01-07", navs, StrategyGenes(vol_target=0.05, min_trade_delta=0.02))
        assert plan["target_exposure"] < plan["current_exposure"]
        assert plan["orders"] and plan["orders"][0]["side"] == SELL

    def test_step_settles_then_submits(self, fees):
        acct = _acct(fees)
        acct.submit_buy("002910", 10000, "2026-01-05")
        result = step(acct, "2026-01-06", self._navs(), StrategyGenes())
        assert len(result["settled"]) == 1
        assert "equity" in result

    def test_portfolio_nav_equal_weight_when_empty(self):
        acct = PaperAccount.create("空", 1000, "2026-01-05")
        navs = {"A": pd.Series([1.0, 1.1, 1.2]), "B": pd.Series([1.0, 0.9, 0.8])}
        out = portfolio_nav(acct, navs)
        assert abs(float(out.iloc[-1]) - 1.0) < 1e-6
