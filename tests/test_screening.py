"""筛选用例、意图路由、组合护栏与相关性诊断测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from agent.intent import (
    IntentRouter, extract_codes, extract_fund_type_from_message,
    extract_return_threshold, extract_round_trips,
)
from analysis import correlation as CORR
from analysis import portfolio as PF
from services.screening import FundCriteria, apply_criteria, csv_escape, to_csv


@pytest.fixture
def df():
    return pd.DataFrame({
        "基金代码": ["000001", "000002", "000003", "000004"],
        "基金简称": ["华夏成长", "国泰金牛", "南方稳利", "易方达蓝筹"],
        "基金类型": ["混合型", "股票型", "债券型", "股票型"],
        "近1月": [-2.5, 3.1, 0.5, 4.2],
        "近3月": [-5.0, 8.5, 1.2, 12.3],
        "近1年": [15.0, 45.0, 8.0, 60.0],
    })


class TestApplyCriteria:
    def test_reports_applied_filters(self, df):
        res = apply_criteria(df, FundCriteria(min_return_1y=20, sort_by="return_1y",
                                             sort_order="desc", limit=10))
        assert len(res.funds) == 2
        assert any(f.startswith("min_return_1y") for f in res.applied)
        assert res.ignored == []

    def test_silently_ignored_filters_are_now_reported(self, df):
        """旧实现会静默吞掉 min_sharpe/min_size，现在必须显式报告。"""
        res = apply_criteria(df, FundCriteria(min_sharpe=1.0, min_size=10))
        assert "min_sharpe" in res.ignored
        assert "min_size" in res.ignored
        payload = res.to_dict()
        assert payload["ignored_filters"] and payload["ignored_reason"]

    def test_size_filter_applies_when_column_exists(self):
        d = pd.DataFrame({"基金代码": ["000001"], "基金简称": ["A"],
                          "基金规模(亿元)": [8.0], "近1年": [10.0]})
        res = apply_criteria(d, FundCriteria(min_size=20))
        assert res.funds == [] and "min_size=20" in res.applied

    def test_type_filter(self, df):
        res = apply_criteria(df, FundCriteria(fund_type="股票"))
        assert len(res.funds) == 2

    def test_ascending(self, df):
        res = apply_criteria(df, FundCriteria(sort_order="asc", limit=4))
        vals = [f["return_1y"] for f in res.funds]
        assert vals == sorted(vals)

    def test_limit(self, df):
        assert len(apply_criteria(df, FundCriteria(limit=2)).funds) == 2

    def test_original_df_not_mutated(self, df):
        before = df["近1年"].tolist()
        apply_criteria(df, FundCriteria(min_return_1y=10))
        assert df["近1年"].tolist() == before


class TestCsv:
    def test_escape(self):
        assert csv_escape(None) == ""
        assert csv_escape("a,b") == '"a,b"'
        assert csv_escape('说"法"') == '"说""法"""'

    def test_to_csv_has_bom_and_preferred_order(self, df):
        res = to_csv(apply_criteria(df, FundCriteria(limit=2)).funds)
        lines = res.splitlines()
        assert lines[0].startswith("﻿")
        header = lines[0].lstrip("﻿").split(",")
        assert header[0] == "code" and header[1] == "name"
        assert "return_1y" in header


class TestIntentRouter:
    def setup_method(self):
        self.router = IntentRouter()

    def test_priority_market_before_ranking(self):
        assert self.router.route("今天市场行情怎么样") == "market"

    def test_ranking(self):
        assert self.router.route("收益排行榜") == "ranking"

    def test_expectancy_intent(self):
        assert self.router.route("000001 手续费多少，划不划算") == "expectancy"

    def test_no_intent(self):
        assert self.router.route("你好") is None

    def test_describe_shape(self):
        d = self.router.describe("对比 000001 000002")
        assert d["intent"] == "compare" and len(d["codes"]) == 2

    def test_extractors(self):
        assert extract_return_threshold("收益超过 20% 的") == 20.0
        assert extract_return_threshold("随便") is None
        assert extract_fund_type_from_message("来点债券型的") == "债券"
        assert extract_codes("000001 和 000002", limit=1) == ["000001"]
        assert extract_round_trips("一年买卖 6 次") == 6.0


class TestPortfolioGuards:
    def test_leverage_guard_clean(self):
        g = PF.leverage_guard({"A": 60.0, "B": 40.0})
        assert g["ok"] and g["short_weights"] == []

    def test_leverage_guard_detects_short(self):
        g = PF.leverage_guard({"A": 130.0, "B": -30.0})
        assert g["ok"] is False and g["short_weights"]

    def test_leverage_guard_detects_leverage(self):
        g = PF.leverage_guard({"A": 0.8, "B": 0.6})   # 小数权重，总敞口 140%
        assert g["ok"] is False
        assert any("杠杆" in n or "敞口" in n for n in g["notes"])

    def test_vol_target_reduces_when_too_volatile(self):
        dates = pd.bdate_range("2023-01-01", periods=120)
        rng = np.random.default_rng(4)
        rets = pd.DataFrame({
            "A": rng.normal(0.0004, 0.02, 120),
            "B": rng.normal(0.0002, 0.018, 120),
        }, index=dates)
        out = PF.vol_target_weights(rets, None, target_vol=0.05)
        assert out["scale"] < 1.0
        assert out["cash_weight_pct"] > 0
        assert abs(sum(out["weights"].values()) + out["cash_weight_pct"] - 100.0) < 0.6

    def test_vol_target_never_adds_leverage(self):
        """即使目标波动率高于已实现波动率，也不加杠杆，只留现金。"""
        dates = pd.bdate_range("2023-01-01", periods=120)
        rng = np.random.default_rng(6)
        rets = pd.DataFrame({"A": rng.normal(0.0004, 0.002, 120),
                             "B": rng.normal(0.0003, 0.002, 120)}, index=dates)
        out = PF.vol_target_weights(rets, None, target_vol=0.30)
        assert out["leveraged"] is False
        assert out["scale"] <= 1.0 + 1e-9
        assert out["guard"]["ok"] is True


class TestOptimizePortfolio:
    def test_blocks_have_guard(self):
        dates = pd.bdate_range("2023-01-01", periods=200)
        rng = np.random.default_rng(9)
        rets = pd.DataFrame({"A": rng.normal(0.0005, 0.012, 200),
                             "B": rng.normal(0.0002, 0.008, 200)}, index=dates)
        res = PF.optimize_portfolio(rets, target_vol=0.10)
        assert all("guard" in b for b in res["results"])
        assert all(b["guard"]["ok"] for b in res["results"])
        assert "vol_targeted" in res


class TestExpectancyHandler:
    """问答里的费后期望必须按用户问到的换手下结论，不能自相矛盾。"""

    def _detail(self):
        from analysis.metrics import risk_metrics
        dates = pd.bdate_range("2023-01-01", periods=250)
        nav = pd.Series(np.cumprod(1 + np.random.default_rng(1).normal(0.0005, 0.01, 250)),
                        index=dates)
        raw = risk_metrics(nav)
        return {
            "code": "000001",
            "info": {"基金简称": "测试基金", "基金类型": "混合型"},
            "history": [{"date": d.strftime("%Y-%m-%d"), "nav": float(v)} for d, v in nav.items()],
            "metrics": {"annual_return": raw["annual_return_pct"],
                        "volatility": raw["volatility_pct"],
                        "sharpe_ratio": raw["sharpe"],
                        "data_points": raw["data_points"]},
        }

    def _reply(self, monkeypatch, message):
        import re
        import agent.rules as R
        detail = self._detail()
        monkeypatch.setattr(R, "fetch_fund_detail", lambda code, ctx: detail)
        from config import load_settings
        ctx = R.AgentContext(settings=load_settings(), repo=None, kb=None)
        text = R.handle_expectancy(message, None, ctx)
        assert text is not None
        m = re.search(r"费后年化 (-?[\d.]+)%", text)
        assert m, text[:200]
        return float(m.group(1)), text

    def test_more_turnover_lower_net(self, monkeypatch):
        low, _ = self._reply(monkeypatch, "000001 一年买卖 1 次划不划算")
        high, htext = self._reply(monkeypatch, "000001 一年买卖 40 次划不划算")
        assert high < low
        assert "频繁申赎" in htext or "惩罚性赎回费" in htext

    def test_mentions_no_short_no_leverage(self, monkeypatch):
        _, text = self._reply(monkeypatch, "000001 划不划算")
        assert "做空" in text and "杠杆" in text


class TestNormalizeCodes:
    """磁盘缓存往返会让 "002910" 变成 2910，出仓必须补零。"""

    def test_int_code_padded(self):
        from sources.normalize import normalize_codes
        df = pd.DataFrame({"基金代码": [2910, 1, 539002], "基金简称": ["a", "b", "c"]})
        out = normalize_codes(df)
        assert list(out["基金代码"]) == ["002910", "000001", "539002"]

    def test_float_code_padded(self):
        from sources.normalize import normalize_codes
        df = pd.DataFrame({"基金代码": [2910.0, 110022.0]})
        assert list(normalize_codes(df)["基金代码"]) == ["002910", "110022"]

    def test_string_code_unchanged(self):
        from sources.normalize import normalize_codes
        df = pd.DataFrame({"基金代码": ["000001", "110022"]})
        assert list(normalize_codes(df)["基金代码"]) == ["000001", "110022"]

    def test_nan_code_stays_empty_not_fake(self):
        """缺失代码不能伪造成 "000000"，宁可留空让前端显示为空。"""
        from sources.normalize import normalize_codes
        df = pd.DataFrame({"基金代码": [np.nan, 1]})
        assert list(normalize_codes(df)["基金代码"]) == ["", "000001"]

    def test_no_code_column_is_noop(self):
        from sources.normalize import normalize_codes
        df = pd.DataFrame({"别的列": [1]})
        assert list(normalize_codes(df).columns) == ["别的列"]


class TestEffectiveNumberOfBets:
    def test_independent_assets_equal_n(self):
        dates = pd.bdate_range("2022-01-01", periods=400)
        rng = np.random.default_rng(2)
        rets = pd.DataFrame({c: rng.normal(0, 0.01, 400) for c in "ABCD"}, index=dates)
        enb = CORR.effective_number_of_bets(CORR.correlation_matrix(rets))
        # 抽样噪声带来的轻微负相关会让 ENB 略高于 N，这是合理的
        assert 3.0 <= enb <= 4.8

    def test_identical_assets_equal_one(self):
        dates = pd.bdate_range("2022-01-01", periods=200)
        x = np.random.default_rng(3).normal(0, 0.01, 200)
        rets = pd.DataFrame({"A": x, "B": x, "C": x}, index=dates)
        assert CORR.effective_number_of_bets(CORR.correlation_matrix(rets)) == pytest.approx(1.0, abs=0.05)
