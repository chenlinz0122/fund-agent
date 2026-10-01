"""
基金筛选 Agent - 单元测试
运行: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from main import (
    resolve_columns, analyze_funds, rule_based_agent,
    ScreenRequest, COLUMN_MAPPING, _extract_return_threshold,
    _extract_fund_type_from_message, MemoryCache,
    search_knowledge, load_knowledge_base, format_knowledge_answer,
    _csv_escape,
)


# ========== 测试数据 ==========
@pytest.fixture
def sample_df():
    """构造测试用 DataFrame"""
    return pd.DataFrame({
        "基金代码": ["000001", "000002", "000003", "000004", "000005"],
        "基金简称": ["华夏成长", "国泰金牛", "南方稳健", "易方达蓝筹", "嘉实沪深300"],
        "基金类型": ["混合型", "股票型", "债券型", "股票型", "指数型"],
        "近1月": [-2.5, 3.1, 0.5, 4.2, 1.8],
        "近3月": [-5.0, 8.5, 1.2, 12.3, 5.6],
        "近6月": [10.0, 25.0, 3.5, 35.0, 15.0],
        "近1年": [15.0, 45.0, 8.0, 60.0, 22.0],
    })


# ========== 列名映射测试 ==========
class TestResolveColumns:
    def test_basic_mapping(self, sample_df):
        col_map = resolve_columns(sample_df)
        assert col_map["code"] == "基金代码"
        assert col_map["name"] == "基金简称"
        assert col_map["type"] == "基金类型"
        assert col_map["return_1y"] == "近1年"

    def test_english_columns(self):
        df = pd.DataFrame({"fund_code": ["000001"], "fund_name": ["Test"], "fund_type": ["股票"]})
        col_map = resolve_columns(df)
        assert col_map["code"] == "fund_code"
        assert col_map["name"] == "fund_name"

    def test_missing_columns(self):
        df = pd.DataFrame({"其他列": [1, 2, 3]})
        col_map = resolve_columns(df)
        assert "code" not in col_map


# ========== 筛选引擎测试 ==========
class TestAnalyzeFunds:
    def test_basic_screening(self, sample_df):
        request = ScreenRequest(min_return_1y=20, sort_by="return_1y", sort_order="desc", limit=10)
        result = analyze_funds(sample_df, request)
        assert len(result) == 3  # 45%、60%、22% 三只满足 >=20
        # 验证排序
        ret_values = [r.get("近1年", r.get("return_1y", 0)) for r in result]
        assert ret_values == sorted(ret_values, reverse=True)

    def test_type_filter(self, sample_df):
        request = ScreenRequest(fund_type="股票", sort_by="return_1y", sort_order="desc", limit=10)
        result = analyze_funds(sample_df, request)
        assert len(result) == 2  # 两只股票型

    def test_limit(self, sample_df):
        request = ScreenRequest(limit=3, sort_by="return_1y", sort_order="desc")
        result = analyze_funds(sample_df, request)
        assert len(result) == 3

    def test_range_filter(self, sample_df):
        request = ScreenRequest(min_return_1y=10, max_return_1y=30, sort_by="return_1y", sort_order="desc", limit=10)
        result = analyze_funds(sample_df, request)
        # 15% 和 22% 在范围内
        assert len(result) == 2

    def test_ascending_sort(self, sample_df):
        request = ScreenRequest(sort_by="return_1y", sort_order="asc", limit=10)
        result = analyze_funds(sample_df, request)
        ret_values = [r.get("近1年", r.get("return_1y", 0)) for r in result]
        assert ret_values == sorted(ret_values)


# ========== 规则引擎测试 ==========
class TestRuleBasedAgent:
    def test_recommend(self, sample_df):
        reply = rule_based_agent("推荐基金", sample_df)
        assert "表现较好" in reply or "基金" in reply

    def test_ranking(self, sample_df):
        reply = rule_based_agent("收益排行榜", sample_df)
        assert "排行榜" in reply or "Top" in reply

    def test_risk(self, sample_df):
        reply = rule_based_agent("稳健型基金", sample_df)
        assert "稳健" in reply or "风险" in reply

    def test_compare_with_codes(self, sample_df, monkeypatch):
        # mock 掉详情抓取，避免测试触网（新的规则引擎在 agent.rules 里调用）
        def mock_detail(code, ctx):
            return {"code": code, "info": {"基金简称": f"基金{code}"},
                    "metrics": {"total_return_1y": 10, "volatility": 5,
                                "sharpe_ratio": 1.5, "max_drawdown": -3},
                    "expectancy": {"net_annual_return_pct": 8.2, "annualized_fee_drag_pct": 1.8}}
        monkeypatch.setattr("agent.rules.fetch_fund_detail", mock_detail)
        reply = rule_based_agent("对比 000001 000002", sample_df)
        assert "对比" in reply
        assert "费后年化" in reply

    def test_default_reply(self, sample_df):
        reply = rule_based_agent("你好", sample_df)
        assert "可以帮你" in reply

    def test_fund_type_query(self, sample_df):
        reply = rule_based_agent("有哪些基金类型", sample_df)
        assert "股票型" in reply
        assert "债券型" in reply


# ========== 参数提取测试 ==========
class TestParameterExtraction:
    def test_extract_return_threshold(self):
        assert _extract_return_threshold("推荐收益超过20%的基金") == 20.0
        assert _extract_return_threshold("收益大于15.5%") == 15.5
        assert _extract_return_threshold("30%以上收益") == 30.0
        assert _extract_return_threshold("推荐基金") is None

    def test_extract_fund_type(self):
        assert _extract_fund_type_from_message("推荐股票型基金") == "股票"
        assert _extract_fund_type_from_message("有什么债券型") == "债券"
        assert _extract_fund_type_from_message("指数型排行") == "指数"
        assert _extract_fund_type_from_message("推荐基金") is None


# ========== 内存缓存测试 ==========
class TestMemoryCache:
    def test_ranking_cache(self):
        mc = MemoryCache(ttl=10)
        assert mc.get_ranking() is None

        df = pd.DataFrame({"a": [1, 2, 3]})
        mc.set_ranking(df)
        cached = mc.get_ranking()
        assert cached is not None
        assert len(cached) == 3

    def test_ranking_cache_expiry(self):
        mc = MemoryCache(ttl=0)  # 立即过期
        df = pd.DataFrame({"a": [1, 2, 3]})
        mc.set_ranking(df)
        import time; time.sleep(0.01)
        assert mc.get_ranking() is None

    def test_detail_cache(self):
        mc = MemoryCache()
        assert mc.get_detail("000001") is None

        mc.set_detail("000001", {"code": "000001", "info": {}})
        assert mc.get_detail("000001") is not None
        assert mc.get_detail("000001")["code"] == "000001"

    def test_invalidate(self):
        mc = MemoryCache()
        mc.set_ranking(pd.DataFrame({"a": [1]}))
        mc.set_detail("000001", {"code": "000001"})
        mc.invalidate_ranking()
        mc.invalidate_detail("000001")
        assert mc.get_ranking() is None
        assert mc.get_detail("000001") is None


# ========== 输入校验测试 ==========
class TestInputValidation:
    def test_screen_request_limit_bounds(self):
        # limit 应在 1-500 范围内
        req = ScreenRequest(limit=1)
        assert req.limit == 1
        req = ScreenRequest(limit=500)
        assert req.limit == 500

    def test_screen_request_limit_too_small(self):
        with pytest.raises(Exception):
            ScreenRequest(limit=0)

    def test_screen_request_limit_too_large(self):
        with pytest.raises(Exception):
            ScreenRequest(limit=501)


# ========== 知识库检索测试 ==========
class TestKnowledgeBase:
    def test_load(self):
        kb = load_knowledge_base()
        assert len(kb) >= 40

    def test_search_sharpe(self):
        results = search_knowledge("什么是夏普比率")
        assert len(results) >= 1
        assert any("夏普" in r.get("title", "") for r in results)

    def test_format_answer_hit(self):
        ans = format_knowledge_answer("基金定投怎么做")
        assert ans is not None
        assert "定投" in ans

    def test_format_answer_miss(self):
        # 无关问题应返回 None，触发上层默认回复
        assert format_knowledge_answer("asdfghjkl") is None


# ========== CSV 转义测试 ==========
class TestCsvEscape:
    def test_plain(self):
        assert _csv_escape("华夏成长") == "华夏成长"

    def test_comma_quoted(self):
        assert _csv_escape("a,b") == '"a,b"'

    def test_none(self):
        assert _csv_escape(None) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
