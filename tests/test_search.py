"""遗传搜索测试：参数盒边界、块级交叉、精英机制、防过拟合复核。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd
import pytest

from analysis.abg import ABGParams
from analysis.search import (
    ABG_SPACE, GENE_BLOCKS, PARAM_SPACE, crossover, evolve, fitness,
    monte_carlo_review, mutate, plateau_check, random_genes,
)
from analysis.signals import Environment, StrategyGenes


def _frame(periods=420, seed=11):
    dates = pd.bdate_range("2021-06-01", periods=periods)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"A": rng.normal(0.0005, 0.012, periods),
                         "B": rng.normal(0.0002, 0.009, periods)}, index=dates)


class TestGeneBox:
    def test_environment_is_not_searchable(self):
        """红线：杠杆上限 / 做空 / 熔断不在参数盒里，进化碰不到。"""
        keys = set(PARAM_SPACE) | set(ABG_SPACE)
        for forbidden in ("max_leverage", "long_only", "global_stop_loss",
                          "execution_lag_days", "one_way_fee", "min_exposure"):
            assert forbidden not in keys

    def test_param_space_covers_genes(self):
        from dataclasses import fields
        searchable = {f.name for f in fields(StrategyGenes)} - {"abg"}
        assert searchable <= set(PARAM_SPACE)

    def test_random_genes_within_box(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            g = random_genes(rng)
            for key, (lo, hi, _is_int) in PARAM_SPACE.items():
                assert lo - 1e-9 <= getattr(g, key) <= hi + 1e-9
            for key, (lo, hi, _is_int) in ABG_SPACE.items():
                assert lo - 1e-9 <= getattr(g.abg, key) <= hi + 1e-9

    def test_random_genes_vary(self):
        rng = np.random.default_rng(2)
        vals = {random_genes(rng).vol_target for _ in range(10)}
        assert len(vals) > 5

    def test_mutate_stays_in_box(self):
        rng = np.random.default_rng(3)
        base = StrategyGenes()
        for _ in range(30):
            g = mutate(base, rate=0.9, sigma_ratio=0.4, rng=rng)
            for key, (lo, hi, _) in PARAM_SPACE.items():
                assert lo - 1e-9 <= getattr(g, key) <= hi + 1e-9


class TestCrossover:
    def test_child_fields_come_from_parents(self):
        rng = np.random.default_rng(4)
        a = random_genes(np.random.default_rng(5))
        b = random_genes(np.random.default_rng(6))
        for _ in range(10):
            child = crossover(a, b, rng)
            for key in PARAM_SPACE:
                assert getattr(child, key) in (getattr(a, key), getattr(b, key))

    def test_blocks_are_defined(self):
        assert {"dynamics", "friction", "timing"} <= set(GENE_BLOCKS)


class TestFitness:
    def test_breakdown_keys(self):
        out = fitness(StrategyGenes(), _frame())
        for key in ("score", "alpha", "cost", "drawdown", "trades", "sharpe"):
            assert key in out

    def test_short_data_penalised(self):
        out = fitness(StrategyGenes(), _frame(30))
        assert out["score"] == pytest.approx(-1e6)

    def test_json_serializable(self):
        json.dumps(fitness(StrategyGenes(), _frame()), ensure_ascii=False)


class TestEvolve:
    def test_small_epoch_structure(self):
        df = _frame()
        report = evolve(df, pop_size=6, generations=2, max_evals=18, seed=99)
        assert "challenger" in report and "champion" in report
        assert report["promotion"]["auto_promoted"] is False
        assert report["n_trials"] == report["evaluations"]
        assert report["n_trials"] > 0
        assert report["environment"]["max_leverage"] == 1.0
        assert len(report["top"]) >= 2
        assert report["final_review"]["available"] is True

    def test_incumbent_beats_itself_only_if_better(self):
        df = _frame()
        genes = StrategyGenes()
        report = evolve(df, pop_size=6, generations=1, max_evals=12,
                        incumbent_genes=genes, seed=5)
        assert report["champion"] is not None
        assert json.dumps(report["champion"]["genes"], ensure_ascii=False)

    def test_respects_eval_cap(self):
        df = _frame()
        report = evolve(df, pop_size=30, generations=20, max_evals=25, seed=1)
        assert report["evaluations"] <= 25 + 30      # 初始种群一次性打满，之后受上限约束


class TestFinalReview:
    def test_monte_carlo_not_in_fitness(self):
        """MC 只在终审出现，fitness 的返回里没有路径统计字段。"""
        keys = set(fitness(StrategyGenes(), _frame()))
        assert "ruin_probability" not in keys

    def test_ruin_probability_bounds(self):
        review = monte_carlo_review(StrategyGenes(), _frame(), n_paths=40, n_trials=10)
        assert 0.0 <= review["ruin_probability"] <= 1.0
        assert review["p05_return_pct"] <= review["median_return_pct"] <= review["p95_return_pct"]
        assert review["deflated_sharpe"]["n_trials"] == 10

    def test_more_trials_lower_dsr(self):
        df = _frame()
        few = monte_carlo_review(StrategyGenes(), df, n_paths=20, n_trials=1)
        many = monte_carlo_review(StrategyGenes(), df, n_paths=20, n_trials=200)
        assert few["deflated_sharpe"]["deflated_sharpe"] >= many["deflated_sharpe"]["deflated_sharpe"]


class TestPlateau:
    def test_returns_interpretation(self):
        out = plateau_check(StrategyGenes(), _frame(), samples=6)
        assert out["available"]
        assert isinstance(out["on_plateau"], bool)
        assert out["center_score"] is not None
        assert "interpretation" in out
