"""模拟引擎测试：无杠杆、无前视、成本与熔断。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from analysis.risk_rules import ExitConfig, no_trade_band
from analysis.simulation import dca_baseline, simulate
from analysis.signals import Environment, StrategyGenes


def _frame(periods=500, seed=5, drift=0.0004, vol=0.012):
    dates = pd.bdate_range("2021-01-04", periods=periods)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(drift * (i + 1), vol, periods)
                         for i, c in enumerate("ABC")}, index=dates)


class TestDcaBaseline:
    def test_rising_nav_positive(self):
        dates = pd.bdate_range("2022-01-01", periods=260)
        nav = pd.Series(np.linspace(1.0, 1.5, 260), index=dates)
        assert dca_baseline(nav) > 10

    def test_flat_nav_zero(self):
        dates = pd.bdate_range("2022-01-01", periods=260)
        assert abs(dca_baseline(pd.Series(1.0, index=dates))) < 1e-6

    def test_short_series(self):
        assert dca_baseline(pd.Series([1.0, 1.1])) == 0.0


class TestSimulate:
    def test_short_data_errors(self):
        assert "error" in simulate(_frame(40))

    def test_exposure_never_exceeds_leverage_cap(self):
        df = _frame()
        res = simulate(df, genes=StrategyGenes(vol_target=0.90))["result"]
        exposures = [row["exposure"] for row in res["strategy"]["exposure_curve"]]
        assert max(exposures) <= 100.0 + 1e-6

    def test_no_lookahead(self):
        """t 日敞口只能由 t-1 及以前的信息决定：改动未来不影响当前敞口。"""
        df = _frame(500)
        cut = 300
        a = simulate(df)["result"]["strategy"]["exposure_curve"]
        shocked = df.copy()
        shocked.iloc[cut:] = shocked.iloc[cut:] + 0.08       # 未来突然暴涨
        b = simulate(shocked)["result"]["strategy"]["exposure_curve"]
        same = [(x["exposure"], y["exposure"]) for x, y in zip(a[:cut - 5], b[:cut - 5])]
        assert all(abs(x - y) < 1e-9 for x, y in same)

    def test_higher_fee_lowers_return(self):
        df = _frame()
        genes = StrategyGenes(min_trade_delta=0.01, rebalance_stride=1)
        cheap = simulate(df, genes=genes, env=Environment(one_way_fee=0.0))["result"]
        pricey = simulate(df, genes=genes, env=Environment(one_way_fee=0.01))["result"]
        assert cheap["strategy"]["metrics"]["total_return_pct"] > \
            pricey["strategy"]["metrics"]["total_return_pct"]
        assert pricey["strategy"]["total_cost_pct"] > 0

    def test_no_trade_band_suppresses_trades(self):
        df = _frame()
        lazy = simulate(df, genes=StrategyGenes(min_trade_delta=0.60, rebalance_stride=1))["result"]
        active = simulate(df, genes=StrategyGenes(min_trade_delta=0.02, rebalance_stride=1))["result"]
        assert lazy["strategy"]["trade_count"] <= active["strategy"]["trade_count"]

    def test_global_stop_loss_fuses(self):
        """持续下跌 + 严熔断：应当触发 fused 并把敞口打到 0。"""
        dates = pd.bdate_range("2022-01-01", periods=400)
        rng = np.random.default_rng(3)
        rets = rng.normal(-0.004, 0.006, 400)
        df = pd.DataFrame({"A": rets, "B": rets * 0.9}, index=dates)
        res = simulate(df, env=Environment(global_stop_loss=0.10),
                       genes=StrategyGenes(vol_target=0.90, accel_gain=0.0,
                                           premium_threshold=1.0, min_trade_delta=0.01,
                                           rebalance_stride=1))["result"]
        assert res["fused"] is True
        assert res["strategy"]["exposure_curve"][-1]["exposure"] <= 0.1

    def test_output_json_serializable(self):
        import json
        res = simulate(_frame())
        json.dumps(res, ensure_ascii=False, default=str)

    def test_guard_reports_environment(self):
        res = simulate(_frame())["result"]
        assert res["guard"]["max_leverage"] == 1.0
        assert res["guard"]["long_only"] is True
        assert res["guard"]["execution_lag_days"] >= 1


class TestNoTradeBand:
    def test_inside_band_no_action(self):
        assert no_trade_band(0.8, 0.85, 0.10) == (0.8, False)

    def test_outside_band_acts(self):
        assert no_trade_band(0.8, 0.5, 0.10) == (0.5, True)

    def test_zero_band_always_acts(self):
        _, changed = no_trade_band(0.8, 0.81, 0.0)
        assert changed is True
