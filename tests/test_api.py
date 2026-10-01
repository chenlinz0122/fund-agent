"""API 集成测试：用一个假仓库替换数据源，验证所有路由的形状与降级行为。

这一组测试是整个分层的证据：路由 -> services -> analysis 全程不触网。
运行: pytest tests/test_api.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main as entry
from agent.intent import IntentRouter
from agent.knowledge import KnowledgeBase
from agent.llm import LLMClient
from config import KNOWLEDGE_FILE, load_settings
from container import AppState

CODES = ["000001", "000002", "000003", "000004"]


def _nav(seed: int, drift: float, periods: int = 900) -> pd.Series:
    dates = pd.bdate_range("2021-01-01", periods=periods)
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumprod(1 + rng.normal(drift, 0.012, periods)), index=dates)


class FakeBreakers:
    def state(self):
        return "closed"

    def state_for(self, key):
        return "closed"

    def open_keys(self):
        return []

    def summary(self):
        return {}


class FakeClient:
    breakers = FakeBreakers()
    calls = 0

    def stats(self):
        return {"calls": 0, "breaker": "closed", "open_breakers": [],
                "breakers": {}, "limiter": {}}


class FakeRankingDisk:
    def exists(self):
        return True

    def age(self):
        return 60.0


class FakeRepository:
    """实现 FundRepository 被路由用到的那部分接口（duck typing）。"""

    ranking_disk = FakeRankingDisk()

    def __init__(self):
        self._nav = {c: _nav(i, 0.0004 * (i + 1)) for i, c in enumerate(CODES)}
        self._bench = _nav(99, 0.0003)
        self._ranking = pd.DataFrame({
            "基金代码": CODES,
            "基金简称": ["华夏成长", "国泰金牛", "南方稳利", "易方达蓝筹"],
            "基金类型": ["混合型", "股票型", "债券型", "股票型"],
            "近1周": [0.3, np.nan, 0.1, np.nan],     # 含 NaN：出参必须能 JSON 化
            "近1月": [-2.5, 3.1, 0.5, 4.2],
            "近3月": [-5.0, 8.5, 1.2, 12.3],
            "近1年": [15.0, 45.0, 8.0, 60.0],
        })
        self._details = {}
        self.calls = []

    def ranking(self, force_refresh: bool = False):
        self.calls.append("ranking")
        return self._ranking.copy()

    def ranking_meta(self):
        return {"memory": {"size": 1}, "disk_age_seconds": 60.0, "degraded": "closed"}

    def ranking_cached(self):
        return True

    def detail_cache_size(self):
        return len(self._details)

    def nav_history(self, code, days=365):
        s = self._nav.get(code)
        return s if s is None or days <= 0 else s.iloc[-days:]

    def benchmark(self, days=365, symbol=None):
        return self._bench if days <= 0 else self._bench.iloc[-days:]

    def basic_info(self, code):
        return {"基金简称": f"基金{code}", "基金类型": "混合型"}

    def cached_detail(self, code):
        return self._details.get(code)

    def cache_detail(self, code, detail):
        self._details[code] = detail

    def aligned_returns(self, codes, days=365):
        navs = {c: self.nav_history(c, days) for c in codes if len(self.nav_history(c, days)) > 1}
        if not navs:
            return pd.DataFrame(), {}
        frame = pd.DataFrame({c: pd.Series(s.values, index=s.index) for c, s in navs.items()})
        return frame.pct_change().dropna(how="any"), navs

    def require_aligned(self, codes, days, min_funds=2):
        from core.errors import InsufficientData
        returns, navs = self.aligned_returns(codes, days)
        if returns.shape[1] < min_funds or len(returns) < 30:
            raise InsufficientData("样本不足")
        return returns, navs

    def news_payload(self):
        return {"news": [{"tag": "电", "summary": "市场平稳", "url": "http://x"}],
                "boards": [{"name": "半导体", "change_pct": 1.2}], "ts": "now"}

    def health(self):
        return {"ranking": {}, "nav": {}}

    def invalidate_all(self):
        self._details.clear()


@pytest.fixture
def client(tmp_path):
    """不进入 with 语句：跳过 lifespan，直接用假仓库构造的 AppState。"""
    from analysis.archive import StrategyArchive
    from paper.store import PaperStore

    settings = load_settings()
    state = AppState(
        settings=settings,
        client=FakeClient(),
        repo=FakeRepository(),
        kb=KnowledgeBase(KNOWLEDGE_FILE),
        llm=LLMClient(settings.llm),
        router=IntentRouter(),
        paper=PaperStore(tmp_path / "paper_accounts.json"),
        archive=StrategyArchive(tmp_path / "strategy_archive.json"),
    )
    entry.app.state.app_state = state
    yield TestClient(entry.app)
    entry.app.state.app_state = None


ALL_PATHS = ["/", "/api/health", "/api/funds", "/api/news", "/api/knowledge",
             "/api/llm/usage", "/api/llm/models", "/api/config"]


class TestSmoke:
    @pytest.mark.parametrize("path", ALL_PATHS)
    def test_get_routes_ok(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 200, resp.text

    def test_frontend_contract_fields(self, client):
        h = client.get("/api/health").json()
        for key in ["status", "llm_configured", "cache_exists", "memory_cache_hit",
                    "detail_cache_size", "knowledge_count", "timestamp"]:
            assert key in h

    def test_funds_returns_records(self, client):
        resp = client.get("/api/funds")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 4 and "基金代码" in body["funds"][0]
        # NaN 必须在出参前变成 null，否则 JSON 编码直接 500
        assert body["funds"][1]["近1周"] is None


class TestScreening:
    def test_screen_applies_and_reports(self, client):
        resp = client.post("/api/screen", json={"min_return_1y": 20, "limit": 10})
        body = resp.json()
        assert body["count"] == 2
        assert any(f.startswith("min_return_1y") for f in body["applied_filters"])

    def test_screen_ignored_filters_visible(self, client):
        body = client.post("/api/screen", json={"min_sharpe": 1.5}).json()
        assert "min_sharpe" in body["ignored_filters"]

    def test_invalid_code_format(self, client):
        resp = client.get("/api/fund/12")
        assert resp.status_code == 400
        assert "6 位" in resp.json()["message"]

    def test_detail_includes_expectancy(self, client):
        body = client.get("/api/fund/000001").json()
        assert body["metrics"]["total_return_1y"] is not None
        e = body["expectancy"]
        assert e["net_annual_return_pct"] <= e["gross_annual_return_pct"]
        assert "annualized_fee_drag_pct" in e

    def test_export_csv(self, client):
        resp = client.get("/api/export?format=csv&limit=3")
        assert resp.status_code == 200
        assert "code" in resp.text.splitlines()[0] or "code" in resp.text.splitlines()[1]


class TestPortfolio:
    def test_analyze(self, client):
        body = client.post("/api/portfolio/analyze",
                           json={"codes": CODES, "days": 365}).json()
        assert len(body["metrics"]) == 4
        assert "correlation" in body and "ranking" in body
        assert body["effective_number_of_bets"] >= 1
        assert abs(sum(body["risk_contributions"].values()) - 100.0) < 1.0

    def test_optimize_has_guards(self, client):
        body = client.post("/api/portfolio/optimize",
                           json={"codes": CODES, "days": 365}).json()
        assert all(b["guard"]["ok"] for b in body["results"])
        assert body["vol_targeted"]["guard"]["ok"]

    def test_backtest_monthly_is_real_rebalance_with_cost(self, client):
        daily = client.post("/api/portfolio/backtest",
                            json={"codes": CODES, "days": 365,
                                  "rebalance": "daily"}).json()
        monthly = client.post("/api/portfolio/backtest",
                              json={"codes": CODES, "days": 365,
                                    "rebalance": "monthly", "cost_pct": 0.3}).json()
        assert monthly["rebalance"] == "monthly"
        assert monthly["total_cost_pct"] >= 0
        # 同一份数据：月末再平衡（含成本）与每日再平衡结果不应完全相同
        assert daily["metrics"]["total_return_pct"] != monthly["metrics"]["total_return_pct"]

    def test_robustness(self, client):
        body = client.post("/api/portfolio/robustness",
                           json={"codes": CODES, "days": 900, "folds": 3,
                                 "train_days": 120, "test_days": 40, "n_trials": 20}).json()
        assert "walk_forward" in body and "per_fund_deflated_sharpe" in body
        assert body["per_fund_deflated_sharpe"][0]["deflated_sharpe"] is not None

    def test_expectancy_batch(self, client):
        body = client.post("/api/portfolio/expectancy",
                           json={"codes": CODES[:2], "round_trips_per_year": 12}).json()
        assert len(body["rows"]) == 2
        assert body["rows"][0]["net_annual_return_pct"] <= body["rows"][0]["gross_annual_return_pct"]

    def test_needs_two_codes(self, client):
        assert client.post("/api/portfolio/analyze", json={"codes": ["000001"]}).status_code == 422


class TestChat:
    def test_rules_engine_used_when_llm_off(self, client):
        body = client.post("/api/chat", json={"message": "收益排行榜", "use_llm": False}).json()
        assert body["engine"] == "rules"
        assert body["intent"] == "ranking"
        assert "排行榜" in body["reply"]

    def test_expectancy_question(self, client):
        body = client.post("/api/chat",
                           json={"message": "000001 管理费一年多少，划不划算", "use_llm": False}).json()
        assert body["intent"] == "expectancy"
        assert "费用" in body["reply"] or "申购费" in body["reply"]

    def test_unknown_falls_back_to_help(self, client):
        body = client.post("/api/chat", json={"message": "asdfghjkl", "use_llm": False}).json()
        assert "可以帮你" in body["reply"]

    def test_intent_probe(self, client):
        body = client.get("/api/intent?message=对比 000001 000002").json()
        assert body["intent"] == "compare"


class TestLLMAndCache:
    def test_usage_shape(self, client):
        body = client.get("/api/llm/usage").json()
        for key in ["llm_enabled", "model", "total_tokens", "budget_tokens",
                    "remaining_tokens", "last_error"]:
            assert key in body

    def test_models_when_disabled(self, client):
        body = client.get("/api/llm/models").json()
        assert body["llm_enabled"] in (True, False)

    def test_budget_validation(self, client):
        assert client.post("/api/llm/budget", json={"budget_tokens": -5}).status_code == 400
        assert client.post("/api/llm/budget", json={"budget_tokens": 100000}).status_code == 200

    def test_budget_guard_stops_llm(self, client):
        state = client.app.state.app_state
        state.llm.cfg = type(state.llm.cfg)(api_key="sk-test", budget_tokens=1)
        state.llm.prompt_tokens, state.llm.completion_tokens = 5, 5
        body = client.post("/api/chat", json={"message": "推荐基金"}).json()
        assert body["engine"] == "rules"
        assert body["degraded_from"] is None or body["degraded_from"] == "llm"

    def test_cache_invalidate(self, client):
        client.get("/api/fund/000002")
        assert client.post("/api/cache/invalidate").status_code == 200


class TestSimulationAndEvolve:
    """新增的模拟/进化接口：重点是环境层红线不能被请求体覆盖。"""

    def test_policy_nested_shape(self, client):
        resp = client.get(f"/api/portfolio/policy?codes={','.join(CODES)}&days=365")
        assert resp.status_code == 200, resp.text
        row = resp.json()["rows"][0]
        assert row["available"] is True
        assert 0.0 <= row["target_exposure"]["exposure"] <= 1.0
        assert "state" in row and "reasons" in row["target_exposure"]

    def test_simulate_exposure_within_cap(self, client):
        resp = client.post("/api/portfolio/simulate",
                           json={"codes": CODES, "days": 365, "season": "春"})
        assert resp.status_code == 200, resp.text
        res = resp.json()["result"]
        assert res["guard"]["max_leverage"] == 1.0
        assert res["aggressiveness"] == 0.80
        for point in res["strategy"]["exposure_curve"]:
            assert point["exposure"] <= 100.0

    def test_environment_fields_are_not_overridable(self, client):
        resp = client.post("/api/portfolio/simulate", json={
            "codes": CODES, "days": 365,
            "genes": {"max_leverage": 5.0, "one_way_fee": 0.0, "vol_target": 0.2}})
        body = resp.json()
        assert "max_leverage" in body["ignored_params"]
        assert "one_way_fee" in body["ignored_params"]
        assert body["result"]["guard"]["max_leverage"] == 1.0
        assert body["policy"]["position_policy"]["vol_target_pct"] == 20.0

    def test_evolve_reports_trials_and_no_auto_promotion(self, client):
        resp = client.post("/api/portfolio/evolve", json={
            "codes": CODES[:3], "days": 365, "pop_size": 6, "generations": 1, "plateau": False})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["n_trials"] == body["evaluations"] and body["n_trials"] > 0
        assert body["promotion"]["auto_promoted"] is False
        assert body["environment"]["max_leverage"] == 1.0
        assert "deflated_sharpe" in body["overfitting_check"]


class TestPaperAndTraining:
    """模拟盘与自训练接口。重点是冻结纪律与"闸门不过不能晋升"。"""

    def _open(self, client, codes=CODES):
        resp = client.post("/api/paper/accounts",
                           json={"name": "测试盘", "capital": 100000,
                                 "codes": codes, "seed_buy": True})
        assert resp.status_code == 200, resp.text
        return resp.json()["account"]["id"]

    def test_open_account_sets_frozen_at(self, client):
        body = client.post("/api/paper/accounts",
                           json={"name": "A", "capital": 50000, "codes": CODES[:2]}).json()
        assert body["frozen_at"]
        assert "样本外" in body["frozen_meaning"]
        assert body["account"]["frozen_at"] == body["frozen_at"]

    def test_seed_buy_creates_pending_orders(self, client):
        aid = self._open(client)
        detail = client.get(f"/api/paper/accounts/{aid}").json()
        assert detail["stats"]["pending_orders"] == len(CODES)
        assert detail["stats"]["cash"] < 100000
        assert "模拟盘不含" in detail["disclaimer"]

    def test_settle_confirms_only_after_submission(self, client):
        aid = self._open(client, codes=[CODES[0]])
        same_day = client.post(f"/api/paper/accounts/{aid}/settle").json()
        assert same_day["confirmed"] == []                     # 当天提交当天不能确认
        later = client.post(f"/api/paper/accounts/{aid}/settle?on_date=2099-01-01").json()
        assert len(later["confirmed"]) == 1
        assert later["confirmed"][0]["fee"] > 0                # 申购费实扣
        detail = client.get(f"/api/paper/accounts/{aid}").json()
        assert detail["positions"][0]["holding_days"] >= 0

    def test_sell_all_and_fee_tier(self, client):
        aid = self._open(client, codes=[CODES[1]])
        client.post(f"/api/paper/accounts/{aid}/settle?on_date=2099-01-01")
        order = client.post(f"/api/paper/accounts/{aid}/orders",
                            json={"code": CODES[1], "side": "sell"}).json()
        assert order["status"] == "pending"
        done = client.post(f"/api/paper/accounts/{aid}/settle?on_date=2099-06-01").json()
        sell = [o for o in done["confirmed"] if o["side"] == "sell"]
        assert sell and sell[0]["fee"] > 0                     # 持有约 5 个月 -> 0.5% 档

    def test_rejects_bad_code_and_insufficient_cash(self, client):
        aid = self._open(client, codes=[CODES[0]])
        assert client.post(f"/api/paper/accounts/{aid}/orders",
                           json={"code": "123", "side": "buy", "amount": 100}).status_code == 400
        big = client.post(f"/api/paper/accounts/{aid}/orders",
                          json={"code": CODES[0], "side": "buy", "amount": 1e9}).json()
        assert big["status"] == "rejected"

    def test_decision_and_auto_endpoints(self, client):
        aid = self._open(client)
        decision = client.get(f"/api/paper/accounts/{aid}/decision")
        assert decision.status_code == 200
        body = decision.json()
        assert 0.0 <= body["target_exposure"] <= 1.0
        auto = client.post(f"/api/paper/accounts/{aid}/auto")
        assert auto.status_code == 200
        assert "settled" in auto.json() and "submitted" in auto.json()

    def test_review_reports_no_out_of_sample_evidence_yet(self, client):
        aid = self._open(client)
        review = client.get(f"/api/train/review?account_id={aid}&narrate=false").json()
        assert review["evidence"] is False
        assert review["facts"]["样本外天数"] == 0
        assert "样本内" in review["note"] or "冻结" in review["note"]
        assert review["narration"]["engine"] == "none"

    def test_missing_account_404(self, client):
        assert client.get("/api/paper/accounts/nope").status_code == 404

    def test_train_round_gates_and_archive(self, client):
        resp = client.post("/api/train/round", json={
            "codes": CODES[:3], "days": 700, "pop_size": 6, "generations": 1,
            "holdout_days": 60, "seed": 7})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["split"]["holdout_days"] == 60
        assert body["split"]["train_days"] >= 250
        assert body["holdout"]["available"] is True
        assert body["record"]["gates"]["passed"] == body["promotable"]
        assert body["evolution"]["evaluations"] > 0

        archive = client.get("/api/train/archive").json()
        assert archive["summary"]["total"] >= 1
        rec = archive["records"][0]
        promote = client.post("/api/train/promote", json={"record_id": rec["id"]})
        if rec["gates"]["passed"]:
            assert promote.json()["ok"] is True
            assert promote.json()["frozen_at"]
        else:
            assert promote.status_code == 400
            assert "闸门" in promote.json()["message"]

    def test_promote_unknown_record_400(self, client):
        resp = client.post("/api/train/promote", json={"record_id": "st-nope"})
        assert resp.status_code == 400


class TestExceptionMapping:
    def test_source_unavailable_maps_503(self, client):
        from core.errors import SourceUnavailable

        state = client.app.state.app_state

        def boom(force_refresh=False):
            raise SourceUnavailable("上游挂了")

        state.repo.ranking = boom
        resp = client.post("/api/screen", json={})
        assert resp.status_code == 503
        assert resp.json()["error"] == "source_unavailable"
