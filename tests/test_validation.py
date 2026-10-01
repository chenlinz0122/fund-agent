"""样本外检验与抗过拟合测试（合成数据，离线）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analysis import validation as V


def _series(dates, drift, vol, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumprod(1 + rng.normal(drift, vol, len(dates))), index=dates)


@pytest.fixture
def navs():
    dates = pd.bdate_range("2019-01-01", periods=760)
    return {c: _series(dates, 0.0002 * (i + 1), 0.011, 10 + i) for i, c in enumerate("ABCDE")}


class TestSharpeSummary:
    def test_basic(self, navs):
        s = V.sharpe_summary(navs["A"].pct_change().dropna())
        assert s["available"] and s["observations"] > 700
        assert s["sharpe_annualized"] == pytest.approx(
            s["sharpe_daily"] * np.sqrt(252), abs=5e-3)

    def test_insufficient(self):
        assert V.sharpe_summary(pd.Series([0.01]))["available"] is False


class TestDeflatedSharpe:
    def test_more_trials_lowers_dsr(self, navs):
        r = navs["A"].pct_change().dropna()
        few = V.deflated_sharpe_ratio(r, n_trials=1)
        many = V.deflated_sharpe_ratio(r, n_trials=500)
        assert few["deflated_sharpe"] > many["deflated_sharpe"]
        assert many["threshold_sharpe_annualized"] > few["threshold_sharpe_annualized"]

    def test_pure_noise_not_robust(self):
        dates = pd.bdate_range("2023-01-01", periods=252)
        r = pd.Series(np.random.default_rng(5).normal(0, 0.01, 252), index=dates)
        out = V.deflated_sharpe_ratio(r, n_trials=200)
        assert out["available"] and out["robust_at_95"] is False

    def test_zero_trials_no_penalty(self, navs):
        out = V.deflated_sharpe_ratio(navs["A"].pct_change().dropna(), n_trials=1)
        assert out["threshold_sharpe_daily"] == 0.0

    def test_expected_max_sharpe_monotonic(self):
        vals = [V.expected_max_sharpe(n, 0.05) for n in (1, 5, 50, 500)]
        assert vals == sorted(vals) and vals[0] == 0.0


class TestInformationCoefficient:
    def test_perfect_rank_agreement(self):
        scores = {"A": 3, "B": 2, "C": 1}
        fwd = {"A": 10, "B": 5, "C": 1}
        out = V.information_coefficient(scores, fwd)
        assert out["available"] and out["rank_ic"] == pytest.approx(1.0)

    def test_too_few_names(self):
        assert V.information_coefficient({"A": 1}, {"A": 2})["available"] is False


class TestWalkForward:
    def test_detects_no_signal_in_random_walks(self, navs):
        out = V.walk_forward(navs, train_days=250, test_days=60, folds=3)
        assert out["available"]
        assert abs(out["mean_rank_ic"]) < 0.35
        assert out["effective_out_of_sample"] is False or out["mean_rank_ic"] > 0.02
        assert len(out["records"]) >= 2

    def test_short_history_returns_unavailable(self, navs):
        short = {c: s.iloc[:120] for c, s in navs.items()}
        out = V.walk_forward(short, train_days=250, test_days=60)
        assert out["available"] is False and "note" in out

    def test_persistent_winner_is_effective(self):
        """构造有持续性的收益：慢涨 A > 中速 B > 高波动 C，样本外 IC 应为正。"""
        dates = pd.bdate_range("2018-01-01", periods=900)
        rng = np.random.default_rng(11)
        drifts = {"A": 0.0016, "B": 0.0008, "C": 0.0, "D": -0.0006, "E": 0.0002}
        navs = {c: pd.Series(np.cumprod(1 + rng.normal(d, 0.006, len(dates))), index=dates)
                for c, d in drifts.items()}
        out = V.walk_forward(navs, train_days=250, test_days=60, folds=6, top_n=2)
        assert out["available"]
        assert out["mean_rank_ic"] > 0.2
