"""agent：问答编排层（意图路由 -> 规则/知识/LLM -> 生成回复）。

依赖方向：api -> agent -> services -> (sources, analysis) -> core。
agent 层不做数值计算，也不直接碰 akshare。
"""
