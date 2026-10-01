"""
量化引擎单元测试（全部使用合成数据，离线运行）
运行: pytest tests/test_quant.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import quant as q


# ========== 测试数据 ==========
@pytest.fixture
def navs():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2023-01-01", periods=300)
    ra = rng.normal(0.001, 0.012, 300)
    rb = rng.normal(0.0003, 0.006, 300)
    rc = ra * 0.85 + rng.normal(0, 0.004, 300)  # 与 A 高相关
    return {
        "A": pd.Series(np.cumprod(1 + ra), index=dates),
        "B": pd.Series(np.cumprod(1 + rb), index=dates),
        "C": pd.Series(np.cumprod(1 + rc), index=dates),
    }


@pytest.fixture
def rets(navs):
    return q.build_returns_matrix(navs)


# ========== 基础指标 ==========
class TestBasicMetrics:
    def test_to_returns_length(self, navs):
        r = q.to_returns(navs["A"])
        assert len(r) == len(navs["A"]) - 1

    def test_to_returns_values(self):
        nav = pd.Series([100.0, 110.0, 99.0])
        r = q.to_returns(nav)
        assert abs(r.iloc[0] - 0.10) < 1e-9
        assert abs(r.iloc[1] - (-0.10)) < 1e-9

    def test_monotonic_nav_no_drawdown(self):
        nav = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        assert q.max_drawdown(nav) == 0.0
        assert q.annualized_return(nav) > 0

    def test_max_drawdown_sign(self, navs):
        mdd = q.max_drawdown(navs["A"])
        assert mdd <= 0  # 返回负值

    def test_annualized_vol_positive(self, navs):
        vol = q.annualized_volatility(q.to_returns(navs["A"]))
        assert vol > 0

    def test_risk_metrics_keys(self, navs):
        m = q.risk_metrics(navs["A"])
        for k in ["annual_return_pct", "volatility_pct", "max_drawdown_pct",
                  "sharpe", "sortino", "calmar", "win_rate_pct", "var_95_pct", "data_points"]:
            assert k in m

    def test_risk_metrics_insufficient_data(self):
        m = q.risk_metrics(pd.Series([1.0]))
        assert "error" in m

    def test_risk_metrics_with_benchmark_has_beta(self, navs):
        m = q.risk_metrics(navs["A"], bench_nav=navs["B"])
        assert "beta" in m and "alpha_pct" in m

    def test_risk_metrics_beta_aligns_by_date(self, navs):
        # 与自身作为基准：beta 应接近 1（验证 risk_metrics 未破坏日期对齐）
        m = q.risk_metrics(navs["A"], bench_nav=navs["A"])
        assert m.get("beta", 0) > 0.9

    def test_var_95_negative_for_normal(self, navs):
        assert q.var_95(q.to_returns(navs["A"])) < 0

    def test_to_returns_preserves_datetime_index(self, navs):
        r = q.to_returns(navs["A"])
        assert isinstance(r.index, pd.DatetimeIndex)

    def test_beta_positive_for_correlated(self, navs):
        # A 与自身高度相关，beta 应显著为正且接近 1
        b, _ = q.beta_alpha(q.to_returns(navs["A"]), q.to_returns(navs["A"]))
        assert b > 0.9

    def test_win_rate_range(self, navs):
        wr = q.win_rate(q.to_returns(navs["A"]))
        assert 0 <= wr <= 100

    def test_types_are_python_float(self, navs):
        # 确保 JSON 可序列化（无 np.float64）
        import json
        m = q.risk_metrics(navs["A"], bench_nav=navs["B"])
        json.dumps(m)  # 不抛异常


# ========== 相关性 ==========
class TestCorrelation:
    def test_diagonal_is_one(self, rets):
        corr = q.correlation_matrix(rets)
        for c in corr.columns:
            assert abs(corr.loc[c, c] - 1.0) < 1e-6

    def test_symmetric(self, rets):
        corr = q.correlation_matrix(rets)
        assert abs(corr.loc["A", "C"] - corr.loc["C", "A"]) < 1e-9

    def test_high_corr_detected(self, rets):
        corr = q.correlation_matrix(rets)
        pairs = q.high_correlation_pairs(corr, threshold=0.5)
        codes = {tuple(sorted([p["a"], p["b"]])) for p in pairs}
        assert ("A", "C") in codes  # A 与 C 人为构造高相关


# ========== 组合优化 ==========
class TestOptimization:
    def _mu_cov(self, rets):
        mu = rets.mean().values * 252
        cov = rets.cov().values * 252
        return mu, cov

    def test_weights_sum_to_one(self, rets):
        mu, cov = self._mu_cov(rets)
        for w in [q.min_variance_weights(cov),
                  q.max_sharpe_weights(mu, cov),
                  q.risk_parity_weights(cov)]:
            assert abs(sum(w) - 1.0) < 1e-6

    def test_long_only_nonnegative(self, rets):
        mu, cov = self._mu_cov(rets)
        for w in [q.min_variance_weights(cov), q.max_sharpe_weights(mu, cov), q.risk_parity_weights(cov)]:
            assert all(x >= -1e-9 for x in w)

    def test_risk_parity_balances_risk(self, rets):
        _, cov = self._mu_cov(rets)
        w = q.risk_parity_weights(cov)
        rc = w * (cov @ w)
        # 各资产风险贡献应大致相等
        assert np.std(rc) / (np.mean(rc) + 1e-12) < 0.35

    def test_optimize_portfolio_structure(self, rets):
        res = q.optimize_portfolio(rets)
        assert "results" in res and "efficient_frontier" in res
        names = {b["name"] for b in res["results"]}
        assert {"max_sharpe", "min_variance", "risk_parity"} <= names
        for b in res["results"]:
            assert abs(sum(b["weights"].values()) - 100.0) < 0.5

    def test_optimize_needs_two_funds(self, rets):
        bad = q.optimize_portfolio(rets[["A"]])
        assert "error" in bad

    def test_frontier_monotonic_return(self, rets):
        mu, cov = self._mu_cov(rets)
        pts = q.efficient_frontier(mu, cov)
        assert len(pts) > 0
        rets_seq = [p["return_pct"] for p in pts]
        assert rets_seq == sorted(rets_seq)  # 前沿点收益递增

    def test_frontier_json_serializable(self, rets):
        import json
        res = q.optimize_portfolio(rets)
        json.dumps(res)


# ========== 回测 ==========
class TestBacktest:
    def test_equal_weight_runs(self, rets):
        bt = q.backtest(rets, rebalance="monthly")
        assert "equity_curve" in bt and "metrics" in bt
        assert len(bt["equity_curve"]) > 0
        assert abs(sum(bt["weights"].values()) - 100.0) < 0.5

    def test_custom_weights_used(self, rets):
        bt = q.backtest(rets, weights={"A": 0.6, "B": 0.2, "C": 0.2}, rebalance="daily")
        assert abs(bt["weights"]["A"] - 60.0) < 0.5

    def test_starting_nav_is_one(self, rets):
        bt = q.backtest(rets, rebalance="none")
        assert abs(bt["equity_curve"][0]["nav"] - 1.0) < 1e-6

    def test_buyhold_vs_rebalance_close_for_equalweight(self, rets):
        b1 = q.backtest(rets, rebalance="daily")["metrics"]["total_return_pct"]
        b2 = q.backtest(rets, rebalance="none")["metrics"]["total_return_pct"]
        # 两者应同数量级（不完全相等）
        assert abs(b1 - b2) < 15

    def test_benchmark_excess_computed(self, rets, navs):
        bench = q.to_returns(navs["B"])
        bt = q.backtest(rets, rebalance="monthly", bench_returns=bench)
        assert "benchmark_return_pct" in bt or "error" in bt
        assert "benchmark_curve" in bt
        import json; json.dumps(bt)

    def test_empty_returns_error(self):
        bt = q.backtest(pd.DataFrame())
        assert "error" in bt


# ========== 多因子打分 ==========
class TestMultiFactor:
    def test_scores_in_range(self, navs):
        mets = []
        for code, nav in navs.items():
            m = q.risk_metrics(nav)
            m["code"] = code
            mets.append(m)
        scored = q.multi_factor_score(mets)
        assert len(scored) == len(navs)
        for s in scored:
            assert 0 <= s["score"] <= 100

    def test_sorted_desc(self, navs):
        mets = []
        for code, nav in navs.items():
            m = q.risk_metrics(nav)
            m["code"] = code
            mets.append(m)
        scored = q.multi_factor_score(mets)
        vals = [s["score"] for s in scored]
        assert vals == sorted(vals, reverse=True)

    def test_dominant_fund_highest_score(self):
        # 构造：A 稳定上涨低波动，B 横盘震荡高波动 → A 各因子全面占优
        dates = pd.bdate_range("2023-01-01", periods=250)
        rng = np.random.default_rng(7)
        nav_a = pd.Series(1.0 * np.cumprod(1 + rng.normal(0.002, 0.002, 250)), index=dates)  # 高收益低波动
        nav_b = pd.Series(1.0 * np.cumprod(1 + rng.normal(0.0, 0.025, 250)), index=dates)     # 0 漂移高波动
        mets = []
        for code, nav in [("A", nav_a), ("B", nav_b)]:
            m = q.risk_metrics(nav)
            m["code"] = code
            mets.append(m)
        scored = q.multi_factor_score(mets)
        assert scored[0]["code"] == "A"

    def test_empty_input(self):
        assert q.multi_factor_score([]) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
