"""期望值层测试：费用侵蚀、收益集中度、结构性负期望来源。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analysis import expectancy as E
from analysis.metrics import risk_metrics, to_returns
from config import AnalysisConfig, FeeConfig


@pytest.fixture
def cfg():
    return FeeConfig()


class TestFeeProfile:
    def test_type_normalization(self, cfg):
        assert E.fee_profile("股票", cfg).fund_type == "股票型"
        assert E.fee_profile("股票型", cfg).management == 1.20
        assert E.fee_profile(None, cfg).fund_type == cfg.default_type
        assert E.fee_profile("不存在的类型", cfg).fund_type == "不存在的型" or \
            E.fee_profile("不存在的类型", cfg).management == E.fee_profile(None, cfg).management

    def test_money_market_has_no_transaction_fee(self, cfg):
        p = E.fee_profile("货币型", cfg)
        assert p.subscription == 0 and p.redemption == 0


class TestFeeDrag:
    def test_annualized_drag_includes_internal_costs(self, cfg):
        p = E.fee_profile("股票型", cfg)
        drag = E.fee_drag(p, holding_years=1.0, round_trips_per_year=1.0)
        # 管理+托管 1.4% + 一次往返 0.65%
        assert abs(drag["annualized_drag_pct"] - (p.annual_internal + 0.65)) < 1e-6

    def test_high_turnover_dominates(self, cfg):
        p = E.fee_profile("股票型", cfg)
        low = E.fee_drag(p, holding_years=1.0, round_trips_per_year=1.0)
        high = E.fee_drag(p, holding_years=1.0, round_trips_per_year=12.0)
        assert high["annualized_drag_pct"] > low["annualized_drag_pct"] * 3

    def test_penalty_redemption_under_7_days(self, cfg):
        p = E.fee_profile("股票型", cfg)
        drag = E.fee_drag(p, holding_years=1.0, round_trips_per_year=60.0,
                          penalty_days=cfg.penalty_days, penalty_redemption=cfg.penalty_redemption)
        assert drag["avg_holding_days"] < cfg.penalty_days
        assert drag["penalty_applies"] is True
        # 60 次往返，每次 0.15% 申购 + 1.5% 惩罚性赎回
        assert abs(drag["transaction_cost_pct"] - 60 * (0.15 + 1.5)) < 1e-6

    def test_no_trips_means_only_internal(self, cfg):
        p = E.fee_profile("债券型", cfg)
        drag = E.fee_drag(p, holding_years=2.0, round_trips_per_year=0)
        assert drag["transaction_cost_pct"] == 0
        assert abs(drag["annualized_drag_pct"] - p.annual_internal) < 1e-6


class TestLuckConcentration:
    def test_detects_one_off_spike(self):
        dates = pd.bdate_range("2023-01-01", periods=250)
        r = np.zeros(250)
        r[100] = 0.20          # 单日暴涨，其余几乎不动
        r += np.random.default_rng(0).normal(0, 0.0005, 250)
        out = E.luck_concentration(pd.Series(r, index=dates))
        assert out["available"] and out["share_of_gain_pct"] > 80

    def test_short_series_unavailable(self):
        out = E.luck_concentration(pd.Series([0.01, 0.02, 0.01]))
        assert out["available"] is False


class TestNegativeExpectancyChecks:
    def test_short_and_leverage_detected(self):
        guard = {"short_weights": [-1.2], "gross_exposure_pct": 130.0}
        items = {c["item"]: c["triggered"] for c in
                 E.negative_expectancy_checks(guard=guard, cfg=AnalysisConfig())}
        assert items["做空"] and items["杠杆"]

    def test_clean_guard_passes(self):
        guard = {"short_weights": [], "gross_exposure_pct": 100.0}
        items = {c["item"]: c["triggered"] for c in
                 E.negative_expectancy_checks(guard=guard, cfg=AnalysisConfig())}
        assert not items["做空"] and not items["杠杆"]

    def test_return_only_ranking_flagged(self):
        items = {c["item"]: c["triggered"] for c in
                 E.negative_expectancy_checks(ranked_by_return_only=True)}
        assert items["只看历史收益排名"]


class TestExpectancyReport:
    def test_fees_can_flip_positive_gross_to_negative_net(self, cfg):
        dates = pd.bdate_range("2023-01-01", periods=252)
        nav = pd.Series(np.cumprod(1 + np.random.default_rng(1).normal(0.00008, 0.011, 252)),
                        index=dates)
        raw = risk_metrics(nav)
        raw["code"] = "X"
        report = E.expectancy_report(raw, fund_type="股票型", returns=to_returns(nav),
                                     round_trips_per_year=60, fee_cfg=cfg,
                                     holding_years=1.0)
        assert report["annualized_fee_drag_pct"] > 0
        assert report["net_annual_return_pct"] < report["gross_annual_return_pct"]
        assert "频繁申赎" in report["risk_of_ruin_notes"] or not report["positive_expectancy"]

    def test_verdict_is_populated(self, cfg):
        dates = pd.bdate_range("2023-01-01", periods=252)
        nav = pd.Series(np.cumprod(1 + np.random.default_rng(2).normal(0.0012, 0.008, 252)),
                        index=dates)
        raw = risk_metrics(nav)
        report = E.expectancy_report(raw, fund_type="指数型", returns=to_returns(nav), fee_cfg=cfg)
        assert report["verdict"]
        assert isinstance(report["positive_expectancy"], bool)
        assert report["assumption"]

    def test_json_serializable(self, cfg):
        import json
        dates = pd.bdate_range("2023-01-01", periods=252)
        nav = pd.Series(np.cumprod(1 + np.random.default_rng(3).normal(0.0009, 0.01, 252)),
                        index=dates)
        report = E.expectancy_report(risk_metrics(nav), fund_type="混合型",
                                     returns=to_returns(nav), fee_cfg=cfg)
        json.dumps(report, ensure_ascii=False)
